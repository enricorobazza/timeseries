from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import shutil
import os

class FileService():
	def __init__(self, folder_id, drive_service):
		self.folder_id = folder_id
		self.files = {}
		self.drive_service = drive_service

	def list_files_in_folder(self):
		results = self.drive_service.files().list(q = "'" + self.folder_id + "' in parents", pageSize=10, fields="nextPageToken, files(id, name)").execute()
		return results.get('files', [])
	

	def download_file_by_id(self, file_id):
		request = self.drive_service.files().get_media(fileId=file_id)
		downloaded = io.BytesIO()
		downloader = MediaIoBaseDownload(downloaded, request)
		done = False
		while done is False:
			_, done = downloader.next_chunk()

		downloaded.seek(0)
		return downloaded.read()

	def download_folder(self, destination):
		files = self.list_files_in_folder()
		for file in files:
			file_id = file.get('id')
			file_name = file.get('name')
			file_path = os.path.join(destination, file_name)
			content = self.download_file_by_id(file_id)
			print("Downloading %s to %s"%(file_name, file_path))
			with open(file_path, "wb") as file:
				file.write(content)
				zip_destination = os.path.join(destination, file_name.replace(".zip", ""))
				if file_path.find(".zip") >= 0:
					shutil.unpack_archive(file_path, zip_destination, "zip")
		
	def sync_file(self, source, destination):
		file_id = None

		if destination in self.files:
			file_id = self.files[destination]

		file_metadata = {
			'name': destination,
			'mimeType': 'text/plain',
			'parents': [self.folder_id]
		}

		media = MediaFileUpload(source, mimetype='text/plain', resumable=True)
		created = None

		if file_id is None:
			created = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
			file_id = created.get('id')
		else:
			del file_metadata["parents"]
			created = self.drive_service.files().update(fileId=file_id, body=file_metadata, media_body=media, fields='id').execute()

		print('File Updated! ID: {}'.format(created.get('id')))

		self.files[destination] = file_id