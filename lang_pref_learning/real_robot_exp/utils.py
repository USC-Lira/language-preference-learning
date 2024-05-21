import re
import cv2


def replay_trajectory_robot(robot, trajectory, speed=1.0):
    """
    Replay a trajectory on the real robot

    :param robot: the robot object
    :param trajectory: the trajectory to replay
    :param speed: the speed factor
    """
    pass



def replay_trajectory_video(traj_images, frame_rate=10):
    """
    Replay a trajectory from a list of images

    :param traj_images: the list of images
    """
    n_frames, h, w, c = traj_images.shape

    # resize the images to a common size

    for i in range(n_frames):
        frame = traj_images[i]

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Current Trajectory', frame)

        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):  # Press 'q' to quit the playback
            break

    cv2.destroyAllWindows()



def remove_special_characters(input_string):
    # Define the pattern to match special characters (non-alphanumeric and non-whitespace)
    pattern = r'[^a-zA-Z0-9\s]'
    # Use re.sub() to replace the special characters with an empty string
    cleaned_string = re.sub(pattern, '', input_string)
    cleaned_string = cleaned_string.strip()
    cleaned_string += "."
    return cleaned_string