select posts, urls, access_time, flag_complete from (
	select posts, urls, access_time, flag_complete, 
	DENSE_RANK () over (partition by urls order by access_time desc) as drank
	from wallstreetbets_posts wp 
)
where drank=2


select *, DENSE_RANK () over (partition by url) as drank
from wallstreetbets_posts_processed wp 
-- =1