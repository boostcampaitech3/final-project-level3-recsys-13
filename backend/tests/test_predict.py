import pytest
from app.api.routes.predictor import return_top10_recipes
from app.schema.prediction import UseridRequest

def test_return_top10_recipes():
    d = return_top10_recipes(UseridRequest(userid=2046))
    assert d.lists == [
        {
            "id": 89204,
            "name": "crock pot chicken with black beans   cream cheese",
            "description": "i love this crock-pot chicken recipe for two reasons: one, my family loves it and two, it is so easy to make! i got this recipe from my sister. she has two children of her own, and they love it too! it's also the best leftovers in the world -- if there are any!"
        },
        {
            "id": 155186,
            "name": "fantastic taco casserole",
            "description": "i originally found this taco casserole recipe in a taste of home magazine.  since then, i have made this recipe at least 50 times.  i love it because the ingredients list isn't complicated and, based on the layers, i can readily remember all ingredients when i am at the store without my recipe card.  i have served this for myself and had wonderful leftovers or many times i have made it for company with a nice green salad to accompany it.  i hope you enjoy it as much as i do."
        },
        {
            "id": 22782,
            "name": "jo mama s world famous spaghetti",
            "description": "my kids will give up a steak dinner for this spaghetti. it is a recipe i have been perfecting for years and it is so good (if i may humbly say) that my kids are disappointed when they eat spaghetti anywhere else but home! in fact they tell me i should open a restaurant and serve only this spaghetti and garlic bread. in response to requests, i have posted the recipe for recipe #28559 that uses approximately 1/2 of the sauce from this \r\nrecipe. have spaghetti one night and lasagna later!  thanks to all of you who have tried my recipe and have written a review.  i read and appreciate every one of them!  chef note:  after i posted this recipe i remembered a funny incident--my dear husband usually has a nice bottle of wine handy so when i make a batch of spaghetti i just help myself to a splash of it. on one occasion, there wasn't a bottle opened, but there was a bottle sitting on the counter so i got out the corkscrew and helped myself. for some reason, the spaghetti that night was the best ever. my husband asked what wine i put in it and i showed him the bottle. he nearly fell off the chair. i had opened a rather expensive bottle he had bought to give his boss. goes to show you--don't use a wine for cooking you wouldn't drink. you get the best results from a good wine!"
        },
        {
            "id": 89207,
            "name": "kittencal s chocolate frosting icing",
            "description": "a wonderful frosting i have used for years made in very little time with perfect results and 100% better than any canned, this frosting is so good you will find yourself eating a fair amount of it before even using it to frost with! --- choose the amount of cocoa powder you desire for either a light, medium or dark frosting, and make certain to sift the cocoa powder and the confectioners sugar before using for the recipe--- this frosting *freezes* very well so double the recipe and freeze one batch for next time --- *note* for an ultra creamy fluffy milk chocolate frosting add in 1 to 1-1/2 cups thawed cool whip topping at the end of mixing the frosting and beat on low speed until blended ------- also see my recipe#80118 ---- recipe#282040 --- recipe#90142"
        },
        {
            "id": 75302,
            "name": "mrs  geraldine s ground beef casserole",
            "description": "this recipe came from a local church fund raising cook book. mrs. geraldine is a good friend of the family. she reminds me so much of mrs. clause, that i can't help but smile every time i think of her. this recipe freezes well. do not bake before freezing. just put it together, freeze it, thaw it and cook."
        },
        {
            "id": 49387,
            "name": "oven fried eggplant  aubergine",
            "description": "this recipe appeared in \"cooking light\" magazine. this is a non-traditional way to prepare eggplant, but it really cuts the calories."
        },
        {
            "id": 95222,
            "name": "pork chops yum yum",
            "description": "i copied this from the orlando sentinel months ago and just now found it. we eat pork chops at least once a week and this is our favorite."
        },
        {
            "id": 27520,
            "name": "poverty meal",
            "description": "when i was a child, my family used to eat this at least once a week due to the fact that it is inexpensive and a little goes a long way. we also added fresh herbs or sometimes whole kernel corn. its even better the next day and it freezes well."
        },
        {
            "id": 8782,
            "name": "roast  sticky  chicken",
            "description": "beautiful and delicious, this incredibly moist roasted chicken puts kenny roger's roasters, boston market, and other rotisserie style chickens to shame! please don't let the word "
        },
        {
            "id": 73166,
            "name": "the best chili you will ever taste",
            "description": "this is the best chili recipe i have ever tried. i'm not sure where the recipe originated, but it is amazing! sometimes, i don't bother adding all four cans of the kidney beans and it still turns out wonderful. once anyone tastes this chili, they will be begging for the recipe!! enjoy!"
        }
    ]