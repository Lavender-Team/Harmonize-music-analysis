import pymysql
from custom_logger import info

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

def is_artist_female_only(music_id):
    with connection.cursor() as cursor:
        sql = (f"SELECT count(*) FROM artist "
               f"WHERE artist_id IN ("
               f"  SELECT artist_id"
               f"  FROM (music m INNER JOIN groups g ON m.group_id = g.group_id) INNER JOIN group_member gm ON g.group_id = gm.group_id"
               f"  WHERE m.music_id = {music_id}) "
               f"AND gender <> 'FEMALE'")
        cursor.execute(sql)

        count = cursor.fetchone()[0]
        return count == 0

def update_music_analysis(music_id, highest_pitch, lowest_pitch, high_pitch_ratio, low_pitch_ratio, pitch_avg, pitch_stat):
    with connection.cursor() as cursor:
        sql = (f"UPDATE music_analysis "
               f"SET status = 'COMPLETE', highest_pitch = {highest_pitch}, lowest_pitch = {lowest_pitch}, "
               f"high_pitch_ratio={high_pitch_ratio}, low_pitch_ratio={low_pitch_ratio}, "
               f'pitch_average={pitch_avg}, pitch_stat="{str(pitch_stat)}" '
               f"WHERE music_id = {music_id}")
        info("[SQL] " + sql)
        cursor.execute(sql)
        connection.commit()
