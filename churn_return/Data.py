from User import User
class Data:
    def __init__(self):
        self.data ={}
        self.data["Users"] = {}


    def get_users(self):
        keys = self.data["Users"].keys()
        return keys

    def get_user(self,user):
        user = User()


    # def get_age(self,age,operator):
    #     keys = self.data["Users"].keys()
    #     people = []
    #     for key in keys:
    #         if operator == "less":
    #             if self.users[key]["age"] < age:
    #                 people += [users[key]]
    #         if operator == "greater":
    #             if self.users[key]["age"] > age:
    #                 people += [users[key]]
    #         if operator == "equal":
    #             if self.users[key]["age"] == age:
    #                 people += [users[key]]
    #     return people

    def set_users(self,users):
        self.data["Users"] = users

    def get_data(self):
        return self.data
    def set_data(self,data):
        self.data = data
