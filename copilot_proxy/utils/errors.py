from typing import *

class FauxPilotException(Exception):
    def __init__(self, message: str, type: Optional[str] = None, param: Optional[str] = None, code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.type = type
        self.param = param
        self.code = code

    def json(self):
        return {
            'error': {
                'message': self.message,
                'type': self.type,
                'param': self.param,
                'code': self.code
            }
        }