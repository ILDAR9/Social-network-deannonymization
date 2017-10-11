var uids = Args.uids.split(",");
var i = 0;

var users = [];

while (i < 12 && i < uids.length){
    var uid = parseInt(uids[i]);
	var personal = API.users.get({"user_id": uid,  "fields" : "sex, city, connections, bdate, contacts"});
	var friends_req = API.friends.get({"user_id": uid});
	users = users + [{"personal": personal, "friends" :friends_req}];
	i = i + 1;
}

return users;


33827906,94665415,12133829,110195469