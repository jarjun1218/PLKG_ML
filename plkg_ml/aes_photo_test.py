# using AES encryption to demo the PLKG system transmitting photo
import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import cv2
import numpy as np

uav_key = "10101010101010101010101010111111111111111111111111110101010101010101010101010100000000000000000000000000"
iot_key = "10101010101010101010101010111111111111111111111111110101010101010101010101010100000000000000000000000000"

# hash the key
uav_key = sha256.sha_byte(uav_key)
print("hashed key:", uav_key)
print("key length:", len(uav_key))

mode = AES.MODE_CBC


# read the photo
image_src = "image/image.png"
image = cv2.imread(image_src)
rows, cols, channels = image.shape
print("Image shape:", image.shape)
cv2.imshow("Image", image)
cv2.waitKey(0)
# cv2.destroyAllWindows()

imageByte = image.tobytes()
print("Image byte length:", len(imageByte))

# padding the image byte
imageBytePadded = pad(imageByte, AES.block_size)
print("Image byte length after padding:", len(imageByte))

# generate the IV
iv = get_random_bytes(AES.block_size)
print("IV:", iv)

# create the cipher
cipher = AES.new(uav_key, mode, iv)

# encrypt the image
cipher_text = cipher.encrypt(imageBytePadded)
print("Cipher text length:", len(cipher_text))

paddedSize = len(imageBytePadded) - len(imageByte)
print("Padded size:", paddedSize)

void = cols * channels - 16 - paddedSize
ivCipher = iv + cipher_text + bytes(void)
imageEncrypted = np.frombuffer(ivCipher, dtype=image.dtype).reshape(rows+1, cols, channels)

# show the encrypted image
cv2.imshow("Encrypted Image", imageEncrypted)
cv2.waitKey(0)
# save the encrypted image
cv2.imwrite("image/image_encrypted.png", imageEncrypted)
# cv2.destroyAllWindows()

# decrypt the image
cipher = AES.new(uav_key, mode, iv)
decryptedPadded = cipher.decrypt(cipher_text)
decrypted = unpad(decryptedPadded, AES.block_size)
print("Decrypted image length:", len(decrypted))

# show the decrypted image
imageDecrypted = np.frombuffer(decrypted, dtype=image.dtype).reshape(rows, cols, channels)
cv2.imshow("Decrypted Image", imageDecrypted)
cv2.waitKey(0)

cv2.destroyAllWindows()