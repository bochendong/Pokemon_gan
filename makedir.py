import os

pokemon_class = ['Zapdos', 'Kadabra', 'Omanyte', 'Shellder', 
'Bellsprout', 'Eevee', 'Jolteon', 'Hypno', 'Seel', 'Zubat', 'Graveler', 'Magneton', 
'Abra', 'Kingler', 'Alakazam', 'Clefable', 'Gyarados', 'Poliwag', 'Rapidash', 'Machamp', 
'Pinsir', 'Muk', 'Seaking', 'Magikarp', 'Goldeen', 'Venusaur', 'Flareon', 'Jigglypuff', 
'Doduo', 'Weedle', 'Vileplume', 'Arcanine', 'Tentacruel', 'Gloom', 'Charmeleon', 'Articuno', 
'Sandshrew', 'Spearow', 'Marowak', 'Snorlax', 'Scyther', 'Primeape', 'Diglett', 
'Onix', 'Mankey', 'Rattata', 'Gengar', 'Gastly', 'Cloyster', 'Weepinbell', 'Dragonair', 
'Squirtle', 'Pikachu', 'Victreebel', 'Charmander', 'Staryu', 'Venonat', 'Vaporeon', 'Ivysaur', 
'Krabby', 'Drowzee', 'Sandslash', 'Kangaskhan', 'Chansey', 'Butterfree', 'Starmie', 'Magmar', 
'Beedrill', 'Ninetales', 'Magnemite', 'Metapod', 'Electrode', 'Raichu', 'Fearow', 'Mewtwo', 
'Kabuto', 'Pidgeotto', 'Blastoise', 'Weezing', 'Golbat', 'Rhyhorn', 
'Moltres', 'Kabutops', 'Aerodactyl', 'Haunter', 'Machop', 'Koffing', 'Pidgeot', 
'Wigglytuff', 'Porygon', 'Vulpix', 'Dugtrio', 'Ditto', 'Raticate', 'Geodude', 'Tentacool', 
'Horsea', 'Oddish', 'Machoke', 'Lapras', 'Poliwrath', 'Slowpoke', 'Bulbasaur', 'Growlithe', 
'Ponyta', 'Parasect', 'Dodrio', 'Meowth', 'Exeggutor', 'Psyduck', 'Tauros', 'Pidgey', 
'Electabuzz', 'Dewgong', 'Wartortle', 'Nidoking', 'Grimer', 'Ekans', 'Caterpie', 'Tangela', 'Kakuna', 
'Golem', 'Slowbro', 'MrMime', 'Jynx', 'Paras', 'Exeggcute', 'Arbok', 'Venomoth', 
'Dratini', 'Cubone', 'Rhydon', 'Dragonite', 'Nidorino', 'Lickitung', 'Nidorina', 'Charizard', 
'Poliwhirl', 'Nidoqueen', 'Farfetchd']


for pokemon in pokemon_class:
    path = "./savedData/" + pokemon
    if(os.path.isdir(path) == False):
        os.mkdir(path)