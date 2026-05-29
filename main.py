import os
os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.security.auth.login.config=/dev/null'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").config("spark.driver.extraJavaOptions", "-Djava.security.auth.login.config=/dev/null").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

os.makedirs("outputs", exist_ok=True)

logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
metadata = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)
joined = logs.join(metadata, on="song_id", how="left")

print("\n=== Task 1: User Favorite Genres ===")
genre_counts = joined.groupBy("user_id", "genre").count()
window_user = Window.partitionBy("user_id").orderBy(col("count").desc())
favorite_genres = (genre_counts.withColumn("rank", rank().over(window_user)).filter(col("rank") == 1).drop("rank").withColumnRenamed("count", "listen_count").orderBy("user_id"))
favorite_genres.show(20, truncate=False)
favorite_genres.write.mode("overwrite").csv("outputs/task1_user_favorite_genres", header=True)

print("\n=== Task 2: Average Listen Time per Genre ===")
avg_listen_time = (joined.groupBy("genre").agg(avg("duration_sec").alias("avg_duration_sec")).orderBy(col("avg_duration_sec").desc()))
avg_listen_time.show(truncate=False)
avg_listen_time.write.mode("overwrite").csv("outputs/task2_avg_listen_time", header=True)

print("\n=== Task 3: Top 10 Genre Loyalty Scores ===")
total_listens = logs.groupBy("user_id").count().withColumnRenamed("count", "total_listens")
top_genre_listens = (genre_counts.withColumn("rank", rank().over(window_user)).filter(col("rank") == 1).drop("rank").withColumnRenamed("count", "top_genre_listens").withColumnRenamed("genre", "top_genre"))
loyalty = (top_genre_listens.join(total_listens, on="user_id").withColumn("loyalty_score", round(col("top_genre_listens") / col("total_listens"), 4)).orderBy(col("loyalty_score").desc()).limit(10))
loyalty.show(truncate=False)
loyalty.write.mode("overwrite").csv("outputs/task3_genre_loyalty_top10", header=True)

print("\n=== Task 4: Users Who Listen Between 12 AM and 5 AM ===")
night_listeners = (logs.withColumn("hour", hour(to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))).filter((col("hour") >= 0) & (col("hour") < 5)).select("user_id", "song_id", "timestamp", "hour").distinct().orderBy("user_id", "timestamp"))
night_listeners.show(20, truncate=False)
night_listeners.write.mode("overwrite").csv("outputs/task4_night_listeners", header=True)

print("\nAll tasks complete. Results saved to outputs/")
spark.stop()
