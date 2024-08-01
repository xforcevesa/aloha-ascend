import cv2
import pickle

for i in range(60):
    with open(f'output/data-{i}.pkl', 'rb') as f:
        classes = pickle.load(f)


    for dd in classes:
        # Print the text
        frame = cv2.putText(dd['frame'], str(dd['follower_pos']), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

