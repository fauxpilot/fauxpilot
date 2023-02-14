from typing import Optional


class FauxPilotException(Exception):
    def __init__(self, message: str, error_type: Optional[str] = None, param: Optional[str] = None,
                 code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

    def json(self):
        return {
            'error': {
                'message': self.message,
                'type': self.error_type,
                'param': self.param,
                'code': self.code
            }
        }
