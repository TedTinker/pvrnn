import smtplib
import os
import zipfile
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def zipdir(path, limit):
    size = 0
    zip_count = 1
    ziph = zipfile.ZipFile('thesis_pics_{}.zip'.format(zip_count), 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(path):
        files.sort()
        for file in files:
            filename = os.path.join(root, file)
            size += os.path.getsize(filename)
            if size > limit:
                ziph.close()
                zip_count += 1
                ziph = zipfile.ZipFile('thesis_pics_{}.zip'.format(zip_count), 'w', zipfile.ZIP_DEFLATED)
                size = os.path.getsize(filename)
            ziph.write(filename)

    ziph.close()
    return zip_count

def send_email(smtp_user, smtp_pass, to_address, subject, body, zip_file):

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_address
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    binary_zip = open(zip_file, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(binary_zip.read())
    encoders.encode_base64(part)

    part.add_header('Content-Disposition', 'attachment; filename= ' + os.path.basename(zip_file))
    msg.attach(part)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(smtp_user, smtp_pass)
    s.sendmail(smtp_user, to_address, msg.as_string())
    s.quit()

os.chdir("maze/saved")
dir_path = "thesis_pics"
zip_limit = 24 * 1024 * 1024  # Limit is 24MB

smtp_user = 'tedjtinker@gmail.com'
smtp_pass = 'iftmktzvjjsmyjpn'
to_address = 'tedjtinker@gmail.com'
subject = 'thesis_pics'
body = 'No text please'

zip_count = zipdir(dir_path, zip_limit)
for i in range(1, zip_count + 1):
    zip_file = 'thesis_pics_{}.zip'.format(i)
    send_email(smtp_user, smtp_pass, to_address, subject + "_{}".format(i), body, zip_file)