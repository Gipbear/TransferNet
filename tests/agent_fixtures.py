"""Shared lightweight clients for checked-batch agent unit tests."""


class FakePathClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.response


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.responses.pop(0)
