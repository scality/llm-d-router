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
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
)

// Parameter key constants for step configuration maps.
const (
	ParamKVConnector = "kv_connector"
	ParamECConnector = "ec_connector"
	// ParamWaitForStoredTimeout, a Go duration string ("5s"), makes the decode
	// step's HTTP request after a prefill carry kv_transfer_params.wait_for_stored
	// with this timeout: the decode worker waits up to that long for the KV the
	// prefill worker is storing, for workers that store the KV after answering
	// the prefill. Requires kv-shared-storage. Unset or 0 disables it.
	ParamWaitForStoredTimeout = "wait_for_stored_timeout"
	// ParamNotifyStoredURL makes the prefill step start the decode step before
	// the prefill response, once the prefill worker reports that the tier it
	// shares with the decode workers holds the prompt KV but a tail. The value
	// is the base URL the prefill workers reach this coordinator replica at
	// (for example "http://${POD_IP}:8080"); ${VAR} references are expanded
	// from the environment. The prefill request carries
	// kv_transfer_params.notify_stored with a per-request URL under it, which
	// the prefill worker POSTs to. Requires kv-shared-storage, HTTP workers,
	// and a decode step with wait_for_stored_timeout. Unset disables it.
	ParamNotifyStoredURL = "notify_stored_url"
	// ParamNotifyStoredTokensLeft and ParamNotifyStoredTokensLeftRatio set
	// that tail: the larger of a token count and a share of the prompt, in
	// [0, 1], so that the decode worker's KV load ends about when the prefill
	// does. Both default to 0: notified once the whole prompt is stored.
	ParamNotifyStoredTokensLeft      = "notify_stored_tokens_left"
	ParamNotifyStoredTokensLeftRatio = "notify_stored_tokens_left_ratio"
)

func buildKVConnector(params map[string]any) (kv.Connector, error) {
	name, err := paramString(params, ParamKVConnector)
	if err != nil {
		return nil, err
	}
	return kv.Build(name)
}

func buildECConnector(params map[string]any) (ec.Connector, error) {
	name, err := paramString(params, ParamECConnector)
	if err != nil {
		return nil, err
	}
	return ec.Build(name)
}

// paramInt reads an integer step parameter. The config decoder may hand a number
// back as int, int64, float64, or json.Number depending on its source and YAML
// representation (a float-formatted literal such as 8192.0 decodes as float64),
// so all are accepted; a non-integral float, an out-of-range value, or a
// non-numeric value is an error rather than a silent truncation or wrap. ok is
// false when the key is absent, leaving the caller's default in place.
func paramInt(params map[string]any, key string) (value int, ok bool, err error) {
	var i64 int64
	switch v := params[key].(type) {
	case nil:
		return 0, false, nil
	case int:
		// An int is by definition in range; return it directly.
		return v, true, nil
	case int64:
		i64 = v
	case float64:
		if v != math.Trunc(v) {
			return 0, false, fmt.Errorf("%s: expected integer, got %v", key, v)
		}
		// float64 cannot represent MaxInt64 exactly (it rounds to 2^63), so
		// reject at that bound to keep the int64(v) conversion well-defined.
		if v < math.MinInt64 || v >= math.MaxInt64 {
			return 0, false, fmt.Errorf("%s: out of range, got %v", key, v)
		}
		i64 = int64(v)
	case json.Number:
		n, convErr := v.Int64()
		if convErr != nil {
			return 0, false, fmt.Errorf("%s: %w", key, convErr)
		}
		i64 = n
	default:
		return 0, false, fmt.Errorf("%s: expected number, got %T", key, v)
	}
	if i64 < math.MinInt || i64 > math.MaxInt {
		return 0, false, fmt.Errorf("%s: out of range for int, got %d", key, i64)
	}
	return int(i64), true, nil
}

// paramDuration reads a duration step parameter from a Go duration string (e.g.
// "30s"). An unparsable string is an error rather than a silent fallback, so a
// malformed value such as "30" (no unit) fails config load instead of running
// the default. ok is false when the key is absent.
func paramDuration(params map[string]any, key string) (value time.Duration, ok bool, err error) {
	switch v := params[key].(type) {
	case nil:
		return 0, false, nil
	case string:
		d, parseErr := time.ParseDuration(v)
		if parseErr != nil {
			return 0, false, fmt.Errorf("%s: %w", key, parseErr)
		}
		return d, true, nil
	default:
		return 0, false, fmt.Errorf("%s: expected duration string, got %T", key, v)
	}
}

// paramFloat reads a number step parameter, accepting the types paramInt
// does. ok is false when the key is absent.
func paramFloat(params map[string]any, key string) (value float64, ok bool, err error) {
	switch v := params[key].(type) {
	case nil:
		return 0, false, nil
	case int:
		return float64(v), true, nil
	case int64:
		return float64(v), true, nil
	case float64:
		return v, true, nil
	case json.Number:
		f, convErr := v.Float64()
		if convErr != nil {
			return 0, false, fmt.Errorf("%s: %w", key, convErr)
		}
		return f, true, nil
	default:
		return 0, false, fmt.Errorf("%s: expected number, got %T", key, v)
	}
}

// paramString reads a string step parameter. A missing key returns an empty
// string; a key present with a non-string value is a configuration error.
func paramString(params map[string]any, key string) (string, error) {
	switch v := params[key].(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	default:
		return "", fmt.Errorf("%s: expected string, got %T", key, v)
	}
}

// paramBool reads a boolean step parameter. A missing key returns ok=false so
// the caller can apply its default; a key present with a non-bool value is a
// configuration error.
func paramBool(params map[string]any, key string) (value bool, ok bool, err error) {
	switch v := params[key].(type) {
	case nil:
		return false, false, nil
	case bool:
		return v, true, nil
	default:
		return false, false, fmt.Errorf("%s: expected bool, got %T", key, v)
	}
}
