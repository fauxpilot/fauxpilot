We offer some ways to connect to the FauxPilot Server. For example, you can create a client by how to open the Openai API, Copilot Plugin, REST API.

## API

Once everything is up and running, you should have a server listening for requests on `http://localhost:5000`. You can now talk to it using the standard [OpenAI API](https://beta.openai.com/docs/api-reference/) (although the full API isn't implemented yet). For example, from Python, using the [OpenAI Python bindings](https://github.com/openai/openai-python):

```python
$ ipython
Python 3.8.10 (default, Mar 15 2022, 12:22:08)
Type 'copyright', 'credits' or 'license' for more information
IPython 8.2.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import openai

In [2]: openai.api_key = 'dummy'

In [3]: openai.api_base = 'http://127.0.0.1:5000/v1'

In [4]: result = openai.Completion.create(model='codegen', prompt='def hello', max_tokens=16, temperature=0.1, stop=["\n\n"])

In [5]: result
Out[5]:
<OpenAIObject text_completion id=cmpl-6hqu8Rcaq25078IHNJNVooU4xLY6w at 0x7f602c3d2f40> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "() {\n    return \"Hello, World!\";\n}"
    }
  ],
  "created": 1659492191,
  "id": "cmpl-6hqu8Rcaq25078IHNJNVooU4xLY6w",
  "model": "codegen",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 15,
    "prompt_tokens": 2,
    "total_tokens": 17
  }
}
```

## Curl with RESTful APIs

```bash
$ curl -s -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"prompt":"def hello","max_tokens":100,"temperature":0.1,"stop":["\n\n"]}' http://localhost:5000/v1/engines/codegen/completions
```

## Copilot Plugin

Perhaps more excitingly, you can configure the official [VSCode Copilot plugin](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) to use your local server. Just edit your `settings.json` to add:

```json
    "github.copilot.advanced": {
        "debug.overrideEngine": "codegen",
        "debug.testOverrideProxyUrl": "http://localhost:5000",
        "debug.overrideProxyUrl": "http://localhost:5000"
    }
```

And you should be able to use Copilot with your own locally hosted suggestions! Of course, probably a lot of stuff is subtly broken. In particular, the probabilities returned by the server are partly fake. Fixing this would require changing FasterTransformer so that it can return log-probabilities for the top k tokens rather that just the chosen token.

Another issue with using the Copilot plugin is that its tokenizer (the component that turns text into a sequence of integers for the model) is slightly different from the one used by CodeGen, so the plugin will sometimes send a request that is longer than CodeGen can handle. You can work around this by replacing the `vocab.bpe` and `tokenizer.json` found in the Copilot extension (something like `.vscode/extensions/github.copilot-[version]/dist/`) with the ones found [here](https://github.com/moyix/fauxpilot/tree/main/copilot_proxy/cgtok/openai_format).

Have fun!

