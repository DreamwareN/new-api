package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSanitizeOpenAIStreamChunk(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		wantChanged   bool
		wantOutput    string
		wantJSONEqual string
	}{
		{
			name:        "strips empty finish_reason empty tool_calls and null function_call",
			input:       `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"finish_reason":"","delta":{"content":"hi","tool_calls":[],"function_call":null}}]}`,
			wantChanged: true,
			wantJSONEqual: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",
				"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		},
		{
			name:          "strips null finish_reason",
			input:         `{"choices":[{"index":0,"finish_reason":null,"delta":{"content":"hi"}}]}`,
			wantChanged:   true,
			wantJSONEqual: `{"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		},
		{
			name:          "strips null tool_calls",
			input:         `{"choices":[{"index":0,"delta":{"tool_calls":null,"content":"hi"}}]}`,
			wantChanged:   true,
			wantJSONEqual: `{"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		},
		{
			name:        "keeps clean chunk untouched",
			input:       `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			wantChanged: false,
			wantOutput:  `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		},
		{
			name:        "keeps real finish reason and tool calls untouched",
			input:       `{"choices":[{"index":0,"finish_reason":"stop","delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"a&b <c>\"}"}}]}}]}`,
			wantChanged: false,
			wantOutput:  `{"choices":[{"index":0,"finish_reason":"stop","delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"a&b <c>\"}"}}]}}]}`,
		},
		{
			name:        "preserves tool call arguments verbatim when junk is removed",
			input:       `{"usage":{"prompt_tokens":1},"choices":[{"index":0,"finish_reason":"","delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\n  \"city\": \"a&b <c>\"\n}"}}],"content":null}}]}`,
			wantChanged: true,
			wantJSONEqual: `{"usage":{"prompt_tokens":1},"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\n  \"city\": \"a&b <c>\"\n}"}}],"content":null}}]}`,
		},
		{
			name:        "returns malformed data unchanged",
			input:       `not a json`,
			wantChanged: false,
			wantOutput:  `not a json`,
		},
		{
			name:        "returns chunk without choices unchanged",
			input:       `{"id":"chatcmpl-1","usage":{"prompt_tokens":1}}`,
			wantChanged: false,
			wantOutput:  `{"id":"chatcmpl-1","usage":{"prompt_tokens":1}}`,
		},
		{
			name:        "keeps unknown top level fields when junk is removed",
			input:       `{"service_tier":"default","choices":[{"index":0,"finish_reason":"","delta":{"content":"hi"}}]}`,
			wantChanged: true,
			wantJSONEqual: `{"service_tier":"default","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := sanitizeOpenAIStreamChunk(tt.input)
			assert.Equal(t, tt.wantChanged, changed)
			if tt.wantOutput != "" {
				assert.Equal(t, tt.wantOutput, got)
				return
			}
			require.JSONEq(t, tt.wantJSONEqual, got)
		})
	}
}
