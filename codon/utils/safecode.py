import random
import string


def safecode(length: int = 4) -> str:
    characters = string.ascii_letters + string.digits
    code = ''.join(random.choices(characters, k=length))
    return code