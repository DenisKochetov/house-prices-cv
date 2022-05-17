# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode



def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    display(PIL.Image.fromarray(im_vis))


def get_building(img, pred):
  pred = pred.copy()
  pred[(pred != 0) * (pred != 1) * (pred != 25)] = 0
  pred[pred != 0] = 1
  pred = pred.reshape(pred.shape[0],pred.shape[1],-1)

  mask = numpy.append(numpy.append(pred,pred, axis = 2), pred, axis = 2).astype(numpy.uint8)
  pred_color = numpy.multiply(img.copy(), mask).astype(numpy.uint8)
  result = PIL.Image.fromarray(pred_color)

  return result



def load_and_norm(path):
    pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    pil_image = PIL.Image.open(path).convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    
    return img_original, output_size, singleton_batch
# Load and normalize one image as a singleton tensor batch
# pil_to_tensor = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
#         std=[0.229, 0.224, 0.225])  # across a large photo dataset.
# ])

colors = scipy.io.loadmat('semantic-segmentation-pytorch/data/color150.mat')['colors']
names = {}
with open('semantic-segmentation-pytorch/data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

  # Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='semantic-segmentation-pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='semantic-segmentation-pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

img_path = 'img/69.png'

def get_building_img(img_path):

    img_original, output_size, singleton_batch =  load_and_norm(img_path)

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    result = get_building(img_original, pred)
    new_img_path = img_path.replace('.png', '_build.png')
    result.save(new_img_path)
    return new_img_path

# img_original, output_size, singleton_batch =  load_and_norm(img_path)


# with torch.no_grad():
#     scores = segmentation_module(singleton_batch, segSize=output_size)
    
# # Get the predicted scores for each pixel
# _, pred = torch.max(scores, dim=1)
# pred = pred.cpu()[0].numpy()

# result = get_building(img_original, pred)
# new_img_path = img_path.replace('.png', '_build.png')
# result.save(new_img_path)
