import pymysql


def get_mysql_password(file_path):
    with open(file_path, 'r') as file:
        password = file.read().strip()
    return password


connection = pymysql.connect(
    host='127.0.0.1',                                   # 데이터베이스 호스트 이름 또는 IP 주소
    user='root',                                        # 데이터베이스 사용자 이름
    password=get_mysql_password('mysql_password.txt'),  # 데이터베이스 사용자 비밀번호
    database='harmonize'                                # 연결할 데이터베이스 이름
)


def close_mysql_connection():
    global connection
    if connection:
        connection.close()


def update_music_analysis(music_id):
    with connection.cursor() as cursor:
        sql = f"UPDATE music_analysis SET status = 'COMPLETE' WHERE music_id = {music_id}"
        cursor.execute(sql)
        connection.commit()
