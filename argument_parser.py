from argparse import ArgumentParser


def parser():
    description = 'Send UDP the result of tracking by dlib'
    argparser = ArgumentParser(description=description)
    argparser.add_argument('-v, --video', type=int, dest='camera', default=0, help='target camera number')
    argparser.add_argument('-a, --address', type=str, dest='address', default='127.0.0.1', help='UDP send address')
    argparser.add_argument('-p, --port', type=int, dest='port', default=6000, help='UDP send port')
    argparser.add_argument('-d', '--display', action='store_true', help='display image')
    argparser.add_argument('-s, --scale', type=float, dest='scale', default=0.5, help='process image scale')
    argparser.add_argument('-k', type=float, dest='k', default=0.2, help='low pass filter weighting factor')
    argparser.add_argument('--color', action='store_false', dest='use_gray_image', help='process color image')
    argparser.add_argument('--debug', action='store_true', help='print send data')
    return argparser.parse_args()


if __name__ == '__main__':
    print(parser())
