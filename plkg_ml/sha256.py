import hashlib

def sha_byte(quantization_result):
    qr = quantization_result.encode('utf-8')
    qr = hashlib.sha256(qr).hexdigest()
    #print(qr)
    byte_result = b''
    for i in range(16):
        two_byte = "0x"+qr[4*i:4*(i+1)]
        byte_result = byte_result + int(two_byte,16).to_bytes(2, byteorder="big")
    return byte_result

