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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/go-logr/logr"
	"github.com/google/uuid"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// pathKVStored is the route a prefill worker POSTs its notify_stored to.
const pathKVStored = "/llm-d/kv-stored/{id}"

// maxNotifyBodyBytes bounds the notify body the route drains.
const maxNotifyBodyBytes = 64 << 10

// storedNotifier holds the notify_stored settings of a prefill step and the
// requests waiting for a notify, keyed by a random id in their URL.
type storedNotifier struct {
	baseURL         string
	tokensLeft      int
	tokensLeftRatio float64

	mu      sync.Mutex
	waiting map[string]chan struct{}
}

// parseStoredNotifier reads the notify_stored parameters; enabled is false
// when notify_stored_url is unset.
func parseStoredNotifier(params map[string]any, kvConn kv.Connector) (notifier *storedNotifier, enabled bool, err error) {
	rawURL, err := paramString(params, ParamNotifyStoredURL)
	if err != nil {
		return nil, false, err
	}
	tokensLeft, hasTokensLeft, err := paramInt(params, ParamNotifyStoredTokensLeft)
	if err != nil {
		return nil, false, err
	}
	ratio, hasRatio, err := paramFloat(params, ParamNotifyStoredTokensLeftRatio)
	if err != nil {
		return nil, false, err
	}
	if rawURL == "" {
		if hasTokensLeft || hasRatio {
			return nil, false, fmt.Errorf("%s and %s require %s", ParamNotifyStoredTokensLeft, ParamNotifyStoredTokensLeftRatio, ParamNotifyStoredURL)
		}
		return nil, false, nil
	}
	if kvConn.Name() != kv.SharedStorage {
		return nil, false, fmt.Errorf("%s requires kv_connector %s, got %s", ParamNotifyStoredURL, kv.SharedStorage, kvConn.Name())
	}
	baseURL := strings.TrimSuffix(os.ExpandEnv(rawURL), "/")
	if u, err := url.Parse(baseURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, false, fmt.Errorf("%s must be an http(s) URL with a host, got %q", ParamNotifyStoredURL, baseURL)
	}
	if tokensLeft < 0 {
		return nil, false, fmt.Errorf("%s must not be negative, got %d", ParamNotifyStoredTokensLeft, tokensLeft)
	}
	if ratio < 0 || ratio > 1 {
		return nil, false, fmt.Errorf("%s must be in [0, 1], got %v", ParamNotifyStoredTokensLeftRatio, ratio)
	}
	return &storedNotifier{
		baseURL:         baseURL,
		tokensLeft:      tokensLeft,
		tokensLeftRatio: ratio,
		waiting:         map[string]chan struct{}{},
	}, true, nil
}

// register returns the notify_stored value of a new request and the channel
// closed when its notify arrives. release forgets the request.
func (n *storedNotifier) register() (params map[string]any, notified <-chan struct{}, release func()) {
	id := uuid.NewString()
	ch := make(chan struct{})
	n.mu.Lock()
	n.waiting[id] = ch
	n.mu.Unlock()
	params = reqcommon.NotifyStoredParams(n.baseURL+strings.Replace(pathKVStored, "{id}", id, 1), n.tokensLeft, n.tokensLeftRatio)
	return params, ch, func() {
		n.mu.Lock()
		delete(n.waiting, id)
		n.mu.Unlock()
	}
}

// serveNotify signals the request the URL names. A notify for an unknown or
// finished request, or a repeated one, is accepted and ignored.
func (n *storedNotifier) serveNotify(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, io.LimitReader(r.Body, maxNotifyBodyBytes))
	id := chi.URLParam(r, "id")
	n.mu.Lock()
	if ch, ok := n.waiting[id]; ok {
		close(ch)
		delete(n.waiting, id)
	}
	n.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

// RegisterRoutes serves the notify_stored route when notify_stored_url is set.
func (s *PrefillStep) RegisterRoutes(r chi.Router) {
	if s.notifier != nil {
		r.Post(pathKVStored, s.notifier.serveNotify)
	}
}

// executeNotified sends the prefill request with notify_stored and returns
// once the notify or the prefill response arrives, whichever is first. On the
// notify, decode starts while the prefill worker stores the rest of the
// prompt KV; the prefill response is then read in the background and only
// logged on failure: neither its body nor its headers reach the request
// context.
func (s *PrefillStep) executeNotified(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, body map[string]any, path string) error {
	notifyParams, notified, release := s.notifier.register()
	defer release()
	kvParams, ok := body[reqcommon.FieldKVTransferParams].(map[string]any)
	if !ok {
		return errors.New("prefill: no kv_transfer_params map to add notify_stored to")
	}
	kvParams[reqcommon.FieldNotifyStored] = notifyParams

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("prefill: marshal: %w", err)
	}
	headers := gatewayHeaders(reqCtx, gateway.PhasePrefill)
	logger.V(logutil.DEFAULT).Info("sending request", "path", path, "notifyStored", true)

	// The prefill request outlives this step when decode starts on the
	// notify; it is cancelled only when the client goes away.
	prefillCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	stop := context.AfterFunc(ctx, cancel)
	results := make(chan prefillResult, 1)
	go func() {
		results <- s.post(prefillCtx, logger, path, bodyBytes, headers)
	}()

	select {
	case res := <-results:
		stop()
		cancel()
		return s.apply(logger, reqCtx, path, res)
	case <-notified:
		logger.V(logutil.DEFAULT).Info("prompt KV stored but its tail, starting decode before the prefill response")
		reqCtx.PrefillDone = true
		go func() {
			res := <-results
			stop()
			cancel()
			if err := errors.Join(res.err, res.decodeErr); err != nil {
				logger.Error(err, "prefill failed after its notify_stored; decode computes the KV it did not store")
			}
		}()
		return nil
	case <-ctx.Done():
		return fmt.Errorf("prefill: %w", ctx.Err())
	}
}
