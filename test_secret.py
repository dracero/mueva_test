from pydantic import SecretStr
key = SecretStr("my_secret_key")
print(type(key))
print(type(key.get_secret_value()))
val = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
print(type(val))
