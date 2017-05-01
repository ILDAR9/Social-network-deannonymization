var group_id = Args.group_id;
var already_crawled_length = parseInt(Args.ofs);

var api_resp = API.groups.getMembers({"group_id": group_id, "sort": "id_asc", "count": 1000, "offset": already_crawled_length});
var members_count = api_resp.count;
var members = api_resp.items;
var offs = 1000; // это сдвиг по участникам группы
already_crawled_length = already_crawled_length + offs;

// пока не получили 20000 и не прошлись по всем участникам
while (offs < 25000 && (offs + already_crawled_length) < members_count){
	// сдвиг участников на offset + мощность массива
 	members = members + API.groups.getMembers({"group_id":  group_id, "sort": "id_asc",
 	 "count": 1000, "offset": (already_crawled_length + offs) }).items;
 	offs = offs + 1000; // увеличиваем сдвиг на 1000
}
return members; // вернуть массив members