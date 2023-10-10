import time
import torch
import torch.nn.functional as F
from torch.autograd import Function

def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++
    Args:

        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


def replace_all_layer_type_recursive(model, old_layer_type, new_layer):
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer
        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)
        

class YOLOV5GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        # self.model(torch.zeros(1, 3, *img_size, device=device))
        # print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits = self.model(input_img)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        return self.forward(input_img)



class YOLOV5GradCAMPlusPlus(YOLOV5GradCAM):
    def __init__(self, model, layer_name, img_size=(640, 640)):
        super(YOLOV5GradCAMPlusPlus, self).__init__(model, layer_name, img_size)
    
    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()
        print("[DEBUG|gradcam.py] cuda memory before FORWARD ", torch.cuda.memory_allocated()/10**8)
        preds, logits = self.model(input_img)
        print("[DEBUG|gradcam.py] cuda memory after FORWARD ", torch.cuda.memory_allocated()/10**8)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value'].squeeze()
            gradients = torch.where(gradients > 0, gradients, 0.)
            indicate = torch.where(gradients > 0, 1., 0.)
            norm_factor = torch.sum(gradients, axis=(1, 2))
            for i in range(len(norm_factor)):
                norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.
            alpha = indicate * norm_factor[:, None, None]
            weights = torch.sum(alpha * gradients, axis=(1, 2))
            activations = self.activations['value'].squeeze()
            k, u, v = gradients.size()
            weights = weights.view(k, 1, 1)
            saliency_map = (weights * activations).sum(0, keepdim=True).unsqueeze(0)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)

            del gradients, indicate, norm_factor, alpha, weights, activations, saliency_map, score, cls, cls_name, logit
            
        # del self.model, self.gradients, self.activations
        print("[DEBUG|gradcam.py] cuda memory after DEL ", torch.cuda.memory_allocated()/10**8)
        return saliency_maps, logits, preds

class GuidedBackpropReLUasModule(torch.nn.Module):
    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input 

class GuidedBackpropReLUModel(YOLOV5GradCAM):
    def __init__(self, model, layer_name, img_size=(640, 640)):
        super(GuidedBackpropReLUModel, self).__init__(model, layer_name, img_size)
    

    def recursive_replace_relu_with_guidedrelu(self, module_top):
        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = torch.nn.ReLU()
        except BaseException:
            pass
    
    def forward(self, input_img, target_class=None):
       # replace ReLU with GuidedBackpropReLU
        
        self.model.zero_grad()
                
                
        # gpu use
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         GuidedBackpropReLUasModule())

        outputs = []
        input_img.requires_grad = True
        preds, logits = self.model(input_img)
        tic = time.time()
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if target_class is None:
                target_class = torch.argmax(logit)
            loss = logit[target_class]
            loss.backward(retain_graph=True)
            output = input_img.grad.cpu().data.squeeze()
            # transpose to (H, W, C)
            output = output.permute(1, 2, 0).unsqueeze(0)
            outputs.append(output)
        # replace GuidedBackpropReLU back with ReLU
        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())
        return outputs, logits, preds
