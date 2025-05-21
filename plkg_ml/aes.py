from Crypto.Cipher import AES
import binascii


def encrypt(plain_text, secret_key):
    """encrypts a plain text string using the given secret_key and nonce - usually a session id"""
    
    nonce = 'j8h6g88uu9ot6r44'
    encobj = AES.new(secret_key, AES.MODE_CBC, nonce)
    
    str_length = len(plain_text) + (16 - (len(plain_text) % 16))
    padded = plain_text.rjust(str_length, b'~')
    
    encrypted_text = encobj.encrypt(padded)
    return encrypted_text.hex()
       
    
def decrypt(encrypted_text, secret_key):
    """decrypts an encrypted string using the given secret_key and nonce - usually a session id"""
    
    
    nonce = 'j8h6g88uu9ot6r44'
    encobj = AES.new(secret_key, AES.MODE_CBC, nonce)
    decrypted_text = encobj.decrypt(binascii.unhexlify(encrypted_text))
    stripped_text = decrypted_text.lstrip(b'~')
    return stripped_text


def byte_check(unKnownTypeText):
    if type(unKnownTypeText) == bytes:
        return unKnownTypeText
    return unKnownTypeText.encode('utf-8')


