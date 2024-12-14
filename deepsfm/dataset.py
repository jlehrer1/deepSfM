import cv2 
from torch.utils.data import Dataset, IterableDataset

class VideoDataset(IterableDataset):
	def __init__(self, video_path, sample_rate):
		self.video_path = video_path
		self.sample_rate = sample_rate
		self.cap = cv2.VideoCapture(video_path)
		self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	def __iter__(self):
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			yield frame

	def __len__(self):
		return self.frame_count
