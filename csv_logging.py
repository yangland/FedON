import csv
import os
from csv import writer

class CsvLogging:
    def __init__(self, file_name, file_location, header):
        self.file_name = file_name
        self.file_location = file_location
        self.fileHeader = header
        self.csv_content = []
        
    def append_csv(self, record):
        self.csv_content.append(record)
    
    def save_csv(self):
        csvFile = open(f'{self.file_location}/{self.file_name}.csv', "w+")
        csvWriter = csv.writer(csvFile, lineterminator='\n')
        csvWriter.writerow(self.fileHeader)
        csvWriter.writerows(self.csv_content)
        csvFile.close()
        
    def append_save_csv(self, record):
        path = f'{self.file_location}/{self.file_name}.csv'
        if os.path.exists(path):
            with open(path, 'a', newline='\n') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(record)
                f_object.close()
        else:
            csvFile = open(path, "w+")
            csv_writer = csv.writer(csvFile, lineterminator='\n')
            csv_writer.writerow(self.fileHeader)
            csv_writer.writerow(record)
            csvFile.close()