/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package steps

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// notifiedPrefill is a prefill step with notify_stored_url served by a test
// coordinator listener, and a gateway whose prefill handler is gw.
type notifiedPrefill struct {
	step *PrefillStep
	gw   *httptest.Server
}

func newNotifiedPrefill(t *testing.T, gw http.HandlerFunc, extra map[string]any) notifiedPrefill {
	t.Helper()
	var routes http.Handler
	coordinator := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routes.ServeHTTP(w, r)
	}))
	t.Cleanup(coordinator.Close)
	t.Setenv("TEST_COORDINATOR_URL", coordinator.URL)

	gwServer := httptest.NewServer(gw)
	t.Cleanup(gwServer.Close)

	params := map[string]any{ParamNotifyStoredURL: "${TEST_COORDINATOR_URL}"}
	for k, v := range extra {
		params[k] = v
	}
	step, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: gwServer.URL}), params)
	if err != nil {
		t.Fatal(err)
	}
	r := chi.NewRouter()
	step.(*PrefillStep).RegisterRoutes(r)
	routes = r
	return notifiedPrefill{step: step.(*PrefillStep), gw: gwServer}
}

func notifyTestRequest() *pipeline.RequestContext {
	return &pipeline.RequestContext{
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "m",
		Body:             map[string]any{"model": "m", "messages": []any{}},
		KVTransferParams: map[string]any{},
	}
}

// readNotifyStored returns the prefill request's kv_transfer_params.notify_stored.
func readNotifyStored(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decode prefill body: %v", err)
		return nil
	}
	kvParams, _ := body[reqcommon.FieldKVTransferParams].(map[string]any)
	notify, _ := kvParams[reqcommon.FieldNotifyStored].(map[string]any)
	if notify == nil {
		t.Errorf("prefill kv_transfer_params has no notify_stored: %v", kvParams)
	}
	return notify
}

func executeAsync(ctx context.Context, step pipeline.Step, reqCtx *pipeline.RequestContext) <-chan error {
	done := make(chan error, 1)
	go func() { done <- step.Execute(ctx, reqCtx) }()
	return done
}

func TestPrefillStep_NotifyStored_ReturnsOnNotify(t *testing.T) {
	release := make(chan struct{})
	answered := make(chan struct{})
	p := newNotifiedPrefill(t, func(w http.ResponseWriter, r *http.Request) {
		defer close(answered)
		notify := readNotifyStored(t, r)
		if notify[reqcommon.FieldTokensLeft] != float64(16) || notify[reqcommon.FieldTokensLeftRatio] != 0.25 {
			t.Errorf("notify_stored = %v, want tokens_left 16 and tokens_left_ratio 0.25", notify)
		}
		url, _ := notify[reqcommon.FieldNotifyStoredURL].(string)
		payload := []byte(`{"request_id":"x","stored_tokens":32,"num_prompt_tokens":40}`)
		resp, err := http.Post(url, "application/json", bytes.NewReader(payload))
		if err != nil {
			t.Errorf("notify: %v", err)
		} else {
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusNoContent {
				t.Errorf("notify status = %d, want 204", resp.StatusCode)
			}
		}
		<-release
		_, _ = w.Write([]byte(`{}`))
	}, map[string]any{ParamNotifyStoredTokensLeft: 16, ParamNotifyStoredTokensLeftRatio: 0.25})

	reqCtx := notifyTestRequest()
	select {
	case err := <-executeAsync(context.Background(), p.step, reqCtx):
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("prefill step did not return on the notify")
	}
	if !reqCtx.PrefillDone {
		t.Fatal("expected PrefillDone once the notify arrived")
	}
	close(release)
	<-answered
}

// A prefill that fails after its notify does not fail the step: decode
// computes the KV the prefill worker did not store.
func TestPrefillStep_NotifyStored_PrefillFailsAfterNotify(t *testing.T) {
	p := newNotifiedPrefill(t, func(w http.ResponseWriter, r *http.Request) {
		notify := readNotifyStored(t, r)
		url, _ := notify[reqcommon.FieldNotifyStoredURL].(string)
		if resp, err := http.Post(url, "application/json", strings.NewReader("{}")); err == nil {
			_ = resp.Body.Close()
		}
		http.Error(w, "boom", http.StatusInternalServerError)
	}, nil)

	reqCtx := notifyTestRequest()
	if err := p.step.Execute(context.Background(), reqCtx); err != nil {
		// The prefill response won the race with the notify.
		t.Skip("prefill response arrived before the notify")
	}
	if !reqCtx.PrefillDone {
		t.Fatal("expected PrefillDone once the notify arrived")
	}
}

func TestPrefillStep_NotifyStored_PrefillAnswerFirst(t *testing.T) {
	p := newNotifiedPrefill(t, func(w http.ResponseWriter, r *http.Request) {
		readNotifyStored(t, r)
		_, _ = w.Write([]byte(`{"kv_transfer_params":{"k":"v"}}`))
	}, nil)

	reqCtx := notifyTestRequest()
	if err := p.step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatal(err)
	}
	if !reqCtx.PrefillDone {
		t.Fatal("expected PrefillDone after the prefill response")
	}
	if reqCtx.KVTransferParams["k"] != "v" {
		t.Fatalf("KVTransferParams = %v, want the prefill response's", reqCtx.KVTransferParams)
	}
}

func TestPrefillStep_NotifyStored_PrefillFailsFirst(t *testing.T) {
	p := newNotifiedPrefill(t, func(w http.ResponseWriter, r *http.Request) {
		readNotifyStored(t, r)
		http.Error(w, "boom", http.StatusInternalServerError)
	}, nil)

	reqCtx := notifyTestRequest()
	if err := p.step.Execute(context.Background(), reqCtx); err == nil {
		t.Fatal("expected the prefill error")
	}
	if reqCtx.PrefillDone {
		t.Fatal("PrefillDone after a failed prefill")
	}
}

func TestPrefillStep_NotifyStored_ClientCancels(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	p := newNotifiedPrefill(t, func(_ http.ResponseWriter, r *http.Request) {
		readNotifyStored(t, r)
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	done := executeAsync(ctx, p.step, notifyTestRequest())
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected an error once the client cancels")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("prefill step did not return on cancellation")
	}
}

// A notify for an unknown or finished request is accepted and ignored.
func TestPrefillStep_NotifyStored_UnknownID(t *testing.T) {
	step, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: "http://unused"}),
		map[string]any{ParamNotifyStoredURL: "http://coordinator:8080"})
	if err != nil {
		t.Fatal(err)
	}
	r := chi.NewRouter()
	step.(*PrefillStep).RegisterRoutes(r)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, strings.Replace(pathKVStored, "{id}", "nope", 1), strings.NewReader("{}")))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want 204", rec.Code)
	}
}

func TestPrefillStep_NotifyStored_Rejects(t *testing.T) {
	tests := []struct {
		name   string
		params map[string]any
	}{
		{name: "another connector", params: map[string]any{ParamKVConnector: "kv-nixl", ParamNotifyStoredURL: "http://c:8080"}},
		{name: "not http", params: map[string]any{ParamNotifyStoredURL: "c:8080"}},
		{name: "negative tokens_left", params: map[string]any{ParamNotifyStoredURL: "http://c:8080", ParamNotifyStoredTokensLeft: -1}},
		{name: "ratio above 1", params: map[string]any{ParamNotifyStoredURL: "http://c:8080", ParamNotifyStoredTokensLeftRatio: 1.5}},
		{name: "tokens_left without url", params: map[string]any{ParamNotifyStoredTokensLeft: 4}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: "http://unused"}), tc.params); err == nil {
				t.Fatalf("expected an error for %v", tc.params)
			}
		})
	}
}

// The route drains a notify body so the connection can be reused.
func TestPrefillStep_NotifyStored_RouteReadsBody(t *testing.T) {
	step, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: "http://unused"}),
		map[string]any{ParamNotifyStoredURL: "http://coordinator:8080"})
	if err != nil {
		t.Fatal(err)
	}
	r := chi.NewRouter()
	step.(*PrefillStep).RegisterRoutes(r)
	body := &countingReader{r: strings.NewReader(`{"stored_tokens":1}`)}
	r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, strings.Replace(pathKVStored, "{id}", "x", 1), body))
	if body.n == 0 {
		t.Fatal("notify body not read")
	}
}

type countingReader struct {
	r io.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}
