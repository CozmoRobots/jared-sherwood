import pyqrcode
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
from PIL import Image
suits =  ["Spades","Diamonds","Clubs","Hearts"]
names = ["Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Jack","Queen","King","Ace"]
deck = []
for suit in suits:
    card = []
    value = 2
    for name in names:
        if name == "Ace":
            card = (suit + " Ace " + str(1) + " " + str(11))
            deck.append(card)
        else:
            card  = (suit +" "+ name +" " + str(value))
            deck.append(card)
            if value != 10:
                value = value + 1
print(deck)
for card in deck:
    qrcode = pyqrcode.create(card)
    qrcode.png(card + ".png", scale  = 8 )
#myText = ('This is a test QRCode')
#qrcode=pyqrcode.create(myText)
#qrcode.png('testqrcode.png',scale = 8) """
