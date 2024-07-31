import LayerMAC as L


class MobileNetV2:

    def __init__(self, model_name, num_classes=1000):
        self.model_name = model_name
        self.num_classes = num_classes

        input_size = (224, 224, 3)  # Height, Width, Channels

        self.num_layers = 10
        self.layers = []

        #layer 1
        num_filters = 32
        stride = 2
        
        self.conv1 = L.Conv2DLayer("Conv2D_1", input_size, num_filters, 3, stride, 1)
        self.layers.append(self.conv1)

        #layer 2
        input_size = self.conv1.calculate_output_size()
        num_filters = 16
        expansion_factor = 1
        n_repeat = 1
        stride = 1

        self.bottleneck1 = L.InvertedResisualBlock("BottleNeck_1", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck1)

        #layer 3
        input_size = self.bottleneck1.calculate_output_size()
        num_filters = 24
        expansion_factor = 6
        n_repeat = 2
        stride = 2

        self.bottleneck2 = L.InvertedResisualBlock("BottleNeck_2", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck2)

        #layer 4
        input_size = self.bottleneck2.calculate_output_size()
        num_filters = 32
        expansion_factor = 6
        n_repeat = 3
        stride = 2

        self.bottleneck3 = L.InvertedResisualBlock("BottleNeck_3", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck3)

        #layer 4
        input_size = self.bottleneck3.calculate_output_size()
        num_filters = 64
        expansion_factor = 6
        n_repeat = 4
        stride = 2

        self.bottleneck4 = L.InvertedResisualBlock("BottleNeck_4", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck4)

        #layer 5
        input_size = self.bottleneck4.calculate_output_size()
        num_filters = 96
        expansion_factor = 6
        n_repeat = 3
        stride = 1

        self.bottleneck5 = L.InvertedResisualBlock("BottleNeck_5", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck5)

        #layer 5
        input_size = self.bottleneck5.calculate_output_size()
        num_filters = 160
        expansion_factor = 6
        n_repeat = 3
        stride = 2

        self.bottleneck6 = L.InvertedResisualBlock("BottleNeck_6", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck6)

        #layer 7
        input_size = self.bottleneck6.calculate_output_size()
        num_filters = 320
        expansion_factor = 6
        n_repeat = 1
        stride = 1

        self.bottleneck7 = L.InvertedResisualBlock("BottleNeck_7", input_size, num_filters, stride, expansion_factor, n_repeat)
        self.layers.append(self.bottleneck7)

        #layer 8
        input_size = self.bottleneck7.calculate_output_size()
        num_filters = 1280
        stride = 1
        
        self.conv2 = L.Conv2DLayer("Conv2D_2", input_size, num_filters, 1, stride, 0)
        self.layers.append(self.conv2)

        #layer 9
        input_size = self.conv2.calculate_output_size()
        stride = 1
        kernel_size = 7
        self.avgpool = L.AvgPooling("AvgPooling", input_size, kernel_size, stride)
        self.layers.append(self.avgpool)

        #layer 10
        input_size = self.avgpool.calculate_output_size()
        num_filters = self.num_classes
        stride = 1
        
        self.conv3 = L.Conv2DLayer("Conv2D_3", input_size, num_filters, 1, stride, 0)
        self.layers.append(self.conv3)

    def __str__(self):
        return self.model_name

    def calculate_macs(self):
        macs = 0
        for layer in self.layers:
            macs += layer.calculate_macs()
        return macs

    def print_model(self):
        for layer in self.layers:
            print(layer)

    def print_stat(self):
        total_macs = self.calculate_macs()

        for layer in self.layers:
            percentage_mac = layer.calculate_macs()/total_macs * 100
            print(f"{layer.layer_name}: {percentage_mac:.2f}%")


model = MobileNetV2("MobileNetV2")


print(f"Number of MMACs: {model.calculate_macs()/1e6:.2f}M")


model.print_stat()

model.print_model()
