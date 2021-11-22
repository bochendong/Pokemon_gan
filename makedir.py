import os

pokemon_class = ['Zapdos', 'Kadabra', 'Alolan Sandslash', 'Omanyte', 'Shellder', 
'Bellsprout', 'Eevee', 'Jolteon', 'Hypno', 'Seel', 'Zubat', 'Graveler', 'Magneton', 
'Abra', 'Kingler', 'Alakazam', 'Clefable', 'Gyarados', 'Poliwag', 'Rapidash', 'Machamp', 
'Pinsir', 'Muk', 'Seaking', 'Magikarp', 'Goldeen', 'Venusaur', 'Flareon', 'Jigglypuff', 
'Doduo', 'Weedle', 'Vileplume', 'Arcanine', 'Tentacruel', 'Gloom', 'Charmeleon', 'Articuno', 
'Sandshrew', 'Spearow', 'Marowak', 'Clefairy', 'Snorlax', 'Scyther', 'Primeape', 'Diglett', 
'Onix', 'Mankey', 'Rattata', 'Voltorb', 'Gengar', 'Gastly', 'Cloyster', 'Weepinbell', 'Dragonair', 
'Squirtle', 'Pikachu', 'Victreebel', 'Charmander', 'Staryu', 'Venonat', 'Vaporeon', 'Ivysaur', 
'Krabby', 'Drowzee', 'Sandslash', 'Kangaskhan', 'Chansey', 'Butterfree', 'Starmie', 'Magmar', 
'Beedrill', 'Ninetales', 'Magnemite', 'Metapod', 'Electrode', 'Raichu', 'Fearow', 'Mewtwo', 
'Kabuto', 'Pidgeotto', 'Hitmonchan', 'Blastoise', 'Weezing', 'Golbat', 'Seadra', 'Rhyhorn', 
'Moltres', 'Golduck', 'Kabutops', 'Aerodactyl', 'Haunter', 'Machop', 'Koffing', 'Pidgeot', 
'Wigglytuff', 'Porygon', 'Vulpix', 'Dugtrio', 'Ditto', 'Raticate', 'Geodude', 'Tentacool', 
'Horsea', 'Oddish', 'Machoke', 'Lapras', 'Poliwrath', 'Omastar', 'Slowpoke', 'Bulbasaur', 'Growlithe', 
'Ponyta', 'Parasect', 'Dodrio', 'Meowth', 'Exeggutor', 'Persian', 'Psyduck', 'Tauros', 'Pidgey', 
'Electabuzz', 'Dewgong', 'Wartortle', 'Nidoking', 'Grimer', 'Ekans', 'Caterpie', 'Tangela', 'Kakuna', 
'Golem', 'Slowbro', 'MrMime', 'Jynx', 'Mew', 'Paras', 'Hitmonlee', 'Exeggcute', 'Arbok', 'Venomoth', 
'Dratini', 'Cubone', 'Rhydon', 'Dragonite', 'Nidorino', 'Lickitung', 'Nidorina', 'Charizard', 
'Poliwhirl', 'Nidoqueen', 'Farfetchd']


for pokemon in pokemon_class:
    path = "./savedData/" + pokemon
    if(os.path.isdir(path) == False):
        os.mkdir(path)