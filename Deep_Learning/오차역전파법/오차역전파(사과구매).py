apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

class AddLayer:

    def __init__(self):
        pass

    def forward(self, x, y):
        return x+y

    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy

class MulLayer:

    def __init__(self):
        self.x = None
        self.t = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out


    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()
add_apple_orange_layer = AddLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward()
price = mul_tax_layer.forward(apple_price, tax)

print(price)