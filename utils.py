def parse_loss_factor(filename):
    f = open(filename)
    iou_scores = {}
    sum = 0
    for line in f.readlines():
        l = line.split(' ')
        iou_scores[l[0]] = float(l[1])
        sum += float(l[1])
    iou_factors = {}
    for k,v in iou_scores.items():
        iou_factors[k] = v/sum
    
    return iou_factors
