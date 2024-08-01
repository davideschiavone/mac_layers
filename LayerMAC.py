import math

def calculate_same_padding(kernel_size):
    padding_top = math.ceil(kernel_size / 2)
    padding_bottom = math.floor(kernel_size / 2)
    return padding_top, padding_bottom

class Layer:
    def __init__(self, layer_name, input_size):
        self.layer_name = layer_name
        self.input_height, self.input_width, self.input_channels = input_size
        self.output_height, self.output_width, self.output_channels = self.calculate_output_size()
        self.num_macs = self.calculate_macs()

    def calculate_output_size(self):
        output_height = self.input_height
        output_width = self.input_width
        output_channels = self.input_channels
        return output_height, output_width, output_channels

    def calculate_macs(self):
        return 0

    def __str__(self):
        return (f"Layer: {self.layer_name}\n"
                f"Input Size: {self.input_height}x{self.input_width}x{self.input_channels}\n"
                f"Output Size: {self.output_height}x{self.output_width}x{self.output_channels}\n"
                f"Number of MMACs: {self.num_macs/1e6:.2f}M")


class Conv2DLayer(Layer):

    def __init__(self, layer_name, input_size, num_filters, kernel_size, stride, padding):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__(layer_name, input_size)
        self.output_height, self.output_width, self.output_channels = self.calculate_output_size()
        self.num_macs = self.calculate_macs()

    def calculate_output_size(self):
        output_height = ((self.input_height - self.kernel_size + 2 * self.padding) // self.stride) + 1
        output_width = ((self.input_width - self.kernel_size + 2 * self.padding) // self.stride) + 1
        output_channels = self.num_filters
        return output_height, output_width, output_channels

    def calculate_filter_size(self):
        filter_height = self.kernel_size
        filter_width = self.kernel_size
        filter_channels = self.input_channels
        filter_number = self.num_filters
        return filter_height, filter_width, filter_channels, filter_number

    def layer_size(self):
        filter_height, filter_width, filter_channels, filter_number = self.calculate_filter_size()
        layer_size = filter_height * filter_width * filter_channels * filter_number
        return layer_size

    def calculate_macs(self):
        output_height, output_width = self.output_height, self.output_width
        macs_per_filter = self.kernel_size * self.kernel_size * self.input_channels
        total_macs = output_height * output_width * macs_per_filter * self.num_filters
        return total_macs

    def __str__(self):
        return (f"Layer: {self.layer_name}\n"
                f"Input Size: {self.input_height}x{self.input_width}x{self.input_channels}\n"
                f"Output Size: {self.output_height}x{self.output_width}x{self.output_channels}\n"
                f"Number of MMACs: {self.num_macs/1e6:.2f}M")

class AvgPooling(Layer):
    def __init__(self, layer_name, input_size, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__(layer_name, input_size)
        self.output_height, self.output_width, self.output_channels = self.calculate_output_size()
        self.num_macs = self.calculate_macs()

    def calculate_output_size(self):
        output_height = ((self.input_height - self.kernel_size) // self.stride) + 1
        output_width = ((self.input_width - self.kernel_size) // self.stride) + 1
        output_channels = self.input_channels
        return output_height, output_width, output_channels

    def layer_size(self):
        return 0

    def calculate_macs(self):
        return 0

    def __str__(self):
        return (f"Layer: {self.layer_name}\n"
                f"Input Size: {self.input_height}x{self.input_width}x{self.input_channels}\n"
                f"Output Size: {self.output_height}x{self.output_width}x{self.output_channels}\n"
                f"Number of MMACs: {self.num_macs/1e6:.2f}M")


class SeparableConv2DLayer(Conv2DLayer):

    def __init__(self, layer_name, input_size, num_filters, kernel_size, stride, padding):
        super().__init__(layer_name, input_size, num_filters, kernel_size, stride, padding)

    def calculate_macs(self):

        output_height, output_width = self.output_height, self.output_width
        macs_per_separable_filter = self.kernel_size * self.kernel_size * 1
        total_separable_macs = output_height * output_width * macs_per_separable_filter * self.num_filters

        total_macs = total_separable_macs
        return total_macs

    def calculate_filter_size(self):
        filter_height = self.kernel_size
        filter_width = self.kernel_size
        filter_channels = 1
        filter_number = self.num_filters
        return filter_height, filter_width, filter_channels, filter_number

class BottleNeck(Layer):

    def __init__(self, layer_name, input_size, num_filters, stride, expansion_factor):

        self.conv1_macs = 0
        self.separablewise_conv_macs = 0
        self.conv2_macs = 0

        self.conv1 = None
        self.separablewise_conv = None
        self.conv2 = None

        self.output_height, self.output_width, self.output_channels = (0, 0, 0)

        super().__init__(layer_name, input_size)

        self.expansion_factor = expansion_factor
 
        padding_depthwise_top, padding_depthwise_bottom = calculate_same_padding(stride)
        self.padding = padding_depthwise_top

        self.conv1 = Conv2DLayer(f"{layer_name}/Conv2D_1_1x1", input_size, self.input_channels * expansion_factor, 1, 1, 0)
        self.conv1_macs = self.conv1.calculate_macs()

        self.separablewise_conv = SeparableConv2DLayer(
            f"{layer_name}/DepthWiseConv2D", 
            self.conv1.calculate_output_size(), 
            self.input_channels * expansion_factor, 
            3, 
            stride, 
            self.padding
        )

        self.separablewise_conv_macs = self.separablewise_conv.calculate_macs()

        self.conv2 = Conv2DLayer(f"{layer_name}/Conv2D_2_1x1", self.separablewise_conv.calculate_output_size(), num_filters, 1, 1, 0)
        self.conv2_macs = self.conv2.calculate_macs()

        self.output_height, self.output_width, self.output_channels = self.conv2.calculate_output_size()
        self.num_macs = self.calculate_macs()

    def calculate_macs(self):
        return self.conv1_macs + self.separablewise_conv_macs + self.conv2_macs
    
    def calculate_output_size(self):
        return self.output_height, self.output_width, self.output_channels

    def layer_size(self):
        if self.conv1 != None and self.separablewise_conv != None and self.conv2 != None:
            return self.conv1.layer_size() + self.separablewise_conv.layer_size() + self.conv2.layer_size()
        else:
            return 0

    def calculate_filter_size(self):
        filter_size = None
        if self.conv1 != None:
            filter_height, filter_width, filter_channels, filter_number = self.conv1.calculate_filter_size()
            filter_size.append((filter_height, filter_width, filter_channels, filter_number))
        else:
            return None
        if self.separablewise_conv_macs != None:
            filter_height, filter_width, filter_channels, filter_number = self.separablewise_conv_macs.calculate_filter_size()
            filter_size.append((filter_height, filter_width, filter_channels, filter_number))
        else:
            return None
        if self.conv2 != None:
            filter_height, filter_width, filter_channels, filter_number = self.conv2.calculate_filter_size()
            filter_size.append((filter_height, filter_width, filter_channels, filter_number))
        else:
            return None
        return filter_size

class InvertedResisualBlock(Layer):
    
    def __init__(self, layer_name, input_size, num_filters, stride, expansion_factor, n_repeat):

        self.layer_name = layer_name
        self.n_repeat = n_repeat
        self.expansion_factor = expansion_factor

        # Initialize the array with None
        self.bottleneck_array = [None] * self.n_repeat
      
        next_input_size = input_size
        for i in range(self.n_repeat):
            if i == 0:
                self.bottleneck_array[i] = BottleNeck(f"{layer_name}/BottleNeck_" + str(i), next_input_size, num_filters, stride, expansion_factor)
            else:
                self.bottleneck_array[i] = BottleNeck(f"{layer_name}/BottleNeck_" + str(i), next_input_size, num_filters, 1, expansion_factor)

            next_input_size = self.bottleneck_array[i].calculate_output_size()

        self.num_macs = 0  
        self.output_height, self.output_width, self.output_channels = self.bottleneck_array[-1].calculate_output_size()
        super().__init__(layer_name, input_size)

    def calculate_macs(self):
        self.num_macs = 0
        for i in range(self.n_repeat):
            self.num_macs += self.bottleneck_array[i].calculate_macs()
        return self.num_macs

    def calculate_output_size(self):
        return self.output_height, self.output_width, self.output_channels


