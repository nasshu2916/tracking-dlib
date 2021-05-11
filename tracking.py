import json
import socket
import sys

import cv2
import dlib
import numpy as np

from argument_parser import parser


def main(args):
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (args.address, args.port)

    prev_shape_2d = None

    # Initialize face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

    model_points = get_model_points()

    cap = cv2.VideoCapture(args.camera)

    print("Start Tracking")
    # Face recognition
    while True:
        # read frame buffer from video
        ret, ori_img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        ori_img = cv2.resize(ori_img, (int(ori_img.shape[1] * args.scale), int(ori_img.shape[0] * args.scale)))

        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY) if args.use_gray_image else ori_img.copy()

        faces = detector(img)

        if len(faces) == 0:
            img_rec = ori_img

        for face in faces:

            # rectangle visualize
            img_rec = cv2.rectangle(ori_img,
                                    pt1=(face.left(), face.top()),
                                    pt2=(face.right(), face.bottom()),
                                    color=(255, 255, 255),
                                    lineType=cv2.LINE_AA,
                                    thickness=2)

            # landmark
            landmark = landmark_predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in landmark.parts()])

            if prev_shape_2d is None:
                prev_shape_2d = np.copy(shape_2d)

            shape_2d = np.array([low_pass_filter(now, prev, args.k) for now, prev in zip(shape_2d, prev_shape_2d)])

            dist_coeffs = np.zeros((4, 1))

            size = img.shape

            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype='double')

            image_points = get_image_points(shape_2d)

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                          image_points,
                                                                          camera_matrix,
                                                                          dist_coeffs,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)

            (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
            mat = np.hstack((rotation_matrix, translation_vector))

            (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)

            direction = get_direction(eulerAngles)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)
            if args.display:
                img_rec = draw_face_landmarks(img_rec, shape_2d)
                img_rec = draw_face_direction(img_rec, image_points, nose_end_point2D, direction)

            # TODO: 認識している顔全て送信するのをやめる
            send_udp(client, server_address, direction)
            if args.debug:
                print(json.dumps(direction))
            prev_shape_2d = np.copy(shape_2d)

        if args.display:
            cv2.imshow('img_rec', img_rec)

            if cv2.waitKey(1) == ord('q'):
                sys.exit(1)


def get_model_points():
    return np.array([
        (0.0, 0.0, 0.0),  # nose
        (0, 0 - 270.0, -90.0),  # jaw
        (-155.0, 145.0, -90.0),  # left edge of eye
        (155.0, 145.0, -90.0),  # right edge of eye
        (-90.0, -125.0, -45.0),  # left edge of mouth
        (90.0, -125.0, -45.0)  # right edge of mouth
    ])


def get_image_points(shape_2d):
    return np.array(
        [
            shape_2d[30],  # nose
            shape_2d[8],  # jaw
            shape_2d[45],  # left edge of eye
            shape_2d[36],  # right edge of eye
            shape_2d[54],  # left edge of mouth
            shape_2d[48]  # right edge of mouth
        ],
        dtype="double")


def get_direction(eulerAngles):
    return {"Pitch": float(eulerAngles[0]), "Yaw": float(eulerAngles[1]), "Roll": float(eulerAngles[2])}


def draw_face_landmarks(img, shape_2d):
    for s in shape_2d:
        cv2.circle(img, (int(s[0]), int(s[1])), 1, (255, 255, 255), -1)
    return img


def draw_face_direction(img, image_points, nose_end_point2D, direction):
    cv2.putText(img, 'Yaw' + str(direction["Yaw"]), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(img, 'Pitch' + str(direction["Pitch"]), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(img, 'Roll' + str(direction["Roll"]), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(img, p1, p2, (255, 0, 0), 2)
    return img


def send_udp(client, server_address, direction):
    message = json.dumps(direction)

    client.sendto(message.encode(), server_address)


def low_pass_filter(now, prev, k):
    return np.array([do_low_pass_filter(x, y, k) for x, y in zip(now, prev)])


def do_low_pass_filter(now, prev, k):
    return (1 - k) * prev + k * now


if __name__ == "__main__":
    args = parser()
    print(args)
    main(args)
