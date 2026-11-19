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

package request

import "time"

// WaitForStoredParams returns the kv_transfer_params.wait_for_stored value of
// a decode request whose prompt KV a prefill pod is storing in a tier both
// pods share: the decode pod waits up to timeout for its whole prompt to be
// stored, then computes what is still missing.
func WaitForStoredParams(timeout time.Duration) map[string]any {
	return map[string]any{FieldTimeoutMS: timeout.Milliseconds()}
}

// NotifyStoredParams returns the kv_transfer_params.notify_stored value of a
// prefill request: the prefill pod POSTs to url once the tier it shares with
// the decode pods holds its prompt KV but the larger of tokensLeft tokens and
// tokensLeftRatio of the prompt.
func NotifyStoredParams(url string, tokensLeft int, tokensLeftRatio float64) map[string]any {
	return map[string]any{
		FieldNotifyStoredURL: url,
		FieldTokensLeft:      tokensLeft,
		FieldTokensLeftRatio: tokensLeftRatio,
	}
}
