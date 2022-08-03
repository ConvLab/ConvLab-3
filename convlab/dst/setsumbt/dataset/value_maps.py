# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convlab3 Unified dataset value maps"""


# MultiWOZ specific label map to avoid duplication and typos in values
VALUE_MAP = {'guesthouse': 'guest house', 'belfry': 'belfray', '-': ' ', '&': 'and', 'b and b': 'bed and breakfast',
             'cityroomz': 'city roomz', '  ': ' ', 'acorn house': 'acorn guest house', 'marriot': 'marriott',
             'worth house': 'the worth house', 'alesbray lodge guest house': 'aylesbray lodge',
             'huntingdon hotel': 'huntingdon marriott hotel', 'huntingd': 'huntingdon marriott hotel',
             'jamaicanchinese': 'chinese', 'barbequemodern european': 'modern european',
             'north americanindian': 'north american', 'caribbeanindian': 'indian', 'sheeps': "sheep's"}


# Domain map for SGD and TM Data
DOMAINS_MAP = {'Alarm_1': 'alarm', 'Banks_1': 'banks', 'Banks_2': 'banks', 'Buses_1': 'bus', 'Buses_2': 'bus',
               'Buses_3': 'bus', 'Calendar_1': 'calendar', 'Events_1': 'events', 'Events_2': 'events',
               'Events_3': 'events', 'Flights_1': 'flights', 'Flights_2': 'flights', 'Flights_3': 'flights',
               'Flights_4': 'flights', 'Homes_1': 'homes', 'Homes_2': 'homes', 'Hotels_1': 'hotel',
               'Hotels_2': 'hotel', 'Hotels_3': 'hotel', 'Hotels_4': 'hotel', 'Media_1': 'media',
               'Media_2': 'media', 'Media_3': 'media', 'Messaging_1': 'messaging', 'Movies_1': 'movies',
               'Movies_2': 'movies', 'Movies_3': 'movies', 'Music_1': 'music', 'Music_2': 'music', 'Music_3': 'music',
               'Payment_1': 'payment', 'RentalCars_1': 'rentalcars', 'RentalCars_2': 'rentalcars',
               'RentalCars_3': 'rentalcars', 'Restaurants_1': 'restaurant', 'Restaurants_2': 'restaurant',
               'RideSharing_1': 'ridesharing', 'RideSharing_2': 'ridesharing', 'Services_1': 'services',
               'Services_2': 'services', 'Services_3': 'services', 'Services_4': 'services', 'Trains_1': 'train',
               'Travel_1': 'travel', 'Weather_1': 'weather', 'movie_ticket': 'movies',
               'restaurant_reservation': 'restaurant', 'coffee_ordering': 'coffee', 'pizza_ordering': 'takeout',
               'auto_repair': 'car_repairs', 'flights': 'flights', 'food-ordering': 'takeout', 'hotels': 'hotel',
               'movies': 'movies', 'music': 'music', 'restaurant-search': 'restaurant', 'sports': 'sports',
               'movie': 'movies'}


# Generic value sets for quantity and time slots
QUANTITIES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10 or more']
TIME = [[(i, j) for i in range(24)] for j in range(0, 60, 5)]
TIME = ['%02i:%02i' % t for l in TIME for t in l]