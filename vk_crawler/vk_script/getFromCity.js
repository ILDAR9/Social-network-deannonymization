var city_id = Args.city_id;
var already_crawled_length = parseInt(Args.ofs);

// делаем первый запрос и создаем массив
var members = API.users.search({"city": city_id, "sort": 0, "count": 1000,
		"offset": already_crawled_length}).items@.id;
// var members = []
// это сдвиг по участникам группы
var offs = 1000;
// пока не получили 20000 и не прошлись по всем участникам
while (offs < 25000){
	// сдвиг участников на offset + мощность массива
 	members = members + API.users.search({"city": city_id, "sort": 0, "count": 1000, 
		"offset": (already_crawled_length + offs) }).items@.id;
	// увеличиваем сдвиг на 1000
 	offs = offs + 1000; 
}
return members; // вернуть массив members