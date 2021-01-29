from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup
from aiogram.types import InlineKeyboardButton

from PIL import Image
from io import BytesIO

import torch

import os
from copy import deepcopy

from face_GAN_man import face_gan_man
from face_GAN_woman import face_gan_woman
from style_transfer import style
from monet_GAN import monet_gan


ERROR_ID = os.environ.get('Error_ID')
API_TOKEN = os.environ.get('API_TOKEN')

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

photo_buffer = {}

class InfoAboutUser:
    def __init__(self):
        self.photos = []


main_menu = InlineKeyboardMarkup()
main_menu.add(InlineKeyboardButton(text='Перенос  стиля с одной картинки на другую',
                                  callback_data='style_transfer'))

main_menu.add(InlineKeyboardButton(text='Поменять пол на фото',
                                  callback_data='change_sex'))
                                  
main_menu.add(InlineKeyboardButton(text='Перенос стиля Моне на изображение',
                                  callback_data='Mone'))
                                       

change_sex_menu = InlineKeyboardMarkup()
change_sex_menu.add(InlineKeyboardButton('Пол на фото мужской',
                                      callback_data='man'))
change_sex_menu.add(InlineKeyboardButton('Пол на фото женский',
                                      callback_data='women'))
change_sex_menu.add(InlineKeyboardButton('Назад',
                                      callback_data='main_menu'))

back = InlineKeyboardMarkup()
back.add(InlineKeyboardButton('Отмена', callback_data='main_menu'))


# start
@dp.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.send_message(message.chat.id,
                           f"Привет, {message.from_user.first_name}!\n" +
			   "Если хочешь подробнее узнать о возможностях бота введи: /help \n" +
			   "Если хочешь задать вопрос разработчику или оставить отзыв введи: /about \n" +
                           "Если хочешь начать,вот что я могу:", reply_markup=main_menu)

    photo_buffer[message.chat.id] = InfoAboutUser()


# help
@dp.message_handler(commands=['help'])
async def send_help(message):
    photo_woman = open('./images/example_women.jpg.jpg', 'rb')
    photo_man = open('./images/example_man.jpg.jpg', 'rb')
    photo_orginal = open('./images/original_image.jpg', 'rb')
    photo_monet = open('./images/like_Monet_style_image.png', 'rb')
    photo_content = open('./images/content_image.jpg', 'rb')
    photo_style = open('./images/style_image.jpg', 'rb')
    photo_result = open('./images/result.jpeg', 'rb')

    await bot.send_message(message.chat.id,
                           "Вот список того,что я могу для тебя сделать:\n"+
			   "Я умею переносить стиль с картинки на картинку.\n" +
			   "Для этого сначала пришли фото на кооторое будем переносить стиль,например:\n")
			   
    await bot.send_photo(message.chat.id, photo_content)

    await bot.send_message(message.chat.id,
                           "Далее пришли фото,стиль с котрого будет взят и перенесен на первое фото:\n")
    await bot.send_photo(message.chat.id, photo_style)

    await bot.send_message(message.chat.id,
                           "После обработки тебе прийдет итоговое фото,которое можно скачть к себе:\n")
    await bot.send_photo(message.chat.id, photo_result)

    await bot.send_message(message.chat.id,
                           "Я умею переносить художественный стиль Моне на изображение.\n" +
			  "Для этого сначала выберите соответсвующую опцию в меню и пришли фото на кооторое будем переносить стиль,например:\n")
    await bot.send_photo(message.chat.id, photo_orginal)

    await bot.send_message(message.chat.id,
                           "После обработки тебе прийдет итоговое фото,которое можно скачть к себе:\n")
    await bot.send_photo(message.chat.id, photo_monet)

    await bot.send_message(message.chat.id,
                           "Я умею менять пол на фотографии на противоположный.\n" +
			  "Для этого сначала выберите соответсвующую опцию в меню, \n" +
			   "далее выбери пол человека изображенного на отправляемом фото,и пришли фото ТОЛЬКО лица для обработки,например если пол мужской:\n")
    await bot.send_photo(message.chat.id, photo_man)

    await bot.send_message(message.chat.id,
                           "Если пол женский:\n")
    await bot.send_photo(message.chat.id, photo_woman)

# about
@dp.message_handler(commands=['about'])
async def send_about(message):
    await bot.send_message(message.chat.id,
                           "Контакты для связи: telegram @Konstanter")
                           
# main menu
@dp.callback_query_handler(lambda c: c.data == 'main_menu')
async def start_menu(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Вот что я могу:")
    await callback_query.message.edit_reply_markup(reply_markup=main_menu)


# change_sex
@dp.callback_query_handler(lambda c: c.data == 'change_sex')
async def change_menu(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Выбери пол человека на фото:")
    await callback_query.message.edit_reply_markup(reply_markup=change_sex_menu)


# style transfer
@dp.callback_query_handler(lambda c: c.data == 'style_transfer')
async def style_transfer(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Пришли изображение, на которое будет перенесен стиль")
    if callback_query.from_user.id not in photo_buffer:
        photo_buffer[callback_query.from_user.id] = InfoAboutUser()

    photo_buffer[callback_query.from_user.id].st_type = 'style_transfer'
    photo_buffer[callback_query.from_user.id].need_photos = 2
    
# Mone
@dp.callback_query_handler(lambda c: c.data == 'Mone')
async def man2women(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text(
        "Пришли фото на которое будет перенесен стиль Mone")

    if callback_query.from_user.id not in photo_buffer:
        photo_buffer[callback_query.from_user.id] = InfoAboutUser()

    photo_buffer[callback_query.from_user.id].st_type = 'Mone'
    photo_buffer[callback_query.from_user.id].need_photos = 1


# to_man
@dp.callback_query_handler(lambda c: c.data == 'women')
async def man2women(callback_query):
    await bot.answer_callback_query(callback_query.id)

    await callback_query.message.edit_text(
        "Отправьте фото лица,пол на вашем фото будет изменен на противоположный")

    if callback_query.from_user.id not in photo_buffer:
        photo_buffer[callback_query.from_user.id] = InfoAboutUser()

    photo_buffer[callback_query.from_user.id].st_type = 'change_sex_M'
    photo_buffer[callback_query.from_user.id].need_photos = 1


# to_women
@dp.callback_query_handler(lambda c: c.data == 'man')
async def women2man(callback_query):
    await bot.answer_callback_query(callback_query.id)

    await callback_query.message.edit_text(
        "Отправьте фото лица,пол на вашем фото будет изменен на противоположный")

    if callback_query.from_user.id not in photo_buffer:
        photo_buffer[callback_query.from_user.id] = InfoAboutUser()

    photo_buffer[callback_query.from_user.id].st_type = 'change_sex_W'
    photo_buffer[callback_query.from_user.id].need_photos = 1


# getting image
@dp.message_handler(content_types=['photo'])
async def get_image(message):
    img = message.photo[-1]

    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)

    if message.chat.id not in photo_buffer:
        await bot.send_message(message.chat.id,
                               "Прежде чем отправлять изображение выберите,что вы хотите с ним сделать", reply_markup=main_menu)
        return

    photo_buffer[message.chat.id].photos.append(photo)

    # style transfer
    if photo_buffer[message.chat.id].st_type == 'style_transfer':
        if photo_buffer[message.chat.id].need_photos == 2:
            photo_buffer[message.chat.id].need_photos = 1

            await bot.send_message(message.chat.id,
                                   "Теперь пришли изображение,стиль с которого будет перенесен на первое изображение",
                                   reply_markup=back)

        elif photo_buffer[message.chat.id].need_photos == 1:
            await bot.send_message(message.chat.id, "Происходит обработка изображения")

            try:
                output = style_transfer(*photo_buffer[message.chat.id].photos)

                await bot.send_document(message.chat.id, deepcopy(output))
                await bot.send_photo(message.chat.id, output)

            except Exception as err:
                await bot.send_message(message.chat.id,
                                   "Произошла ошибка. Сообщение об ошибке отправлено создателю бота.")
                await bot.send_message(ERROR_ID, "Произошла ошибка: " + str(err))
		
            await bot.send_message(message.chat.id,
                                   "Что будем делать дальше?", reply_markup=main_menu)		           			
            del photo_buffer[message.chat.id]

    # change sex on foto woman
    elif photo_buffer[message.chat.id].st_type == 'change_sex_W' and \
            photo_buffer[message.chat.id].need_photos == 1:
        await bot.send_message(message.chat.id, "Происходит обработка изображения")

        try:
            output = face_woman(photo_buffer[message.chat.id].photos[0])
            await bot.send_message(message.chat.id, "Смена пола на женский")
            await bot.send_document(message.chat.id, deepcopy(output))		
            await bot.send_photo(message.chat.id, output)
         
        except Exception as err:
            await bot.send_message(message.chat.id,
                                   "Произошла ошибка. Сообщение об ошибке отправлено создателю бота.")
            await bot.send_message(ERROR_ID, "Произошла ошибка: " + str(err))
		
        await bot.send_message(message.chat.id,
                               "Что будем делать дальше?", reply_markup=main_menu)
        del photo_buffer[message.chat.id]
        
        
    # change sex on foto man
    elif photo_buffer[message.chat.id].st_type == 'change_sex_M' and \
            photo_buffer[message.chat.id].need_photos == 1:
        await bot.send_message(message.chat.id, "Происходит обработка изображения")

        try:
            output = face_man(photo_buffer[message.chat.id].photos[0])
            await bot.send_message(message.chat.id, "Смена пола на мужской")
            await bot.send_document(message.chat.id, deepcopy(output))		
            await bot.send_photo(message.chat.id, output)
		
        except Exception as err:
            await bot.send_message(message.chat.id,
                                   "Произошла ошибка. Сообщение об ошибке отправлено создателю бота." )

            await bot.send_message(ERROR_ID, "Произошла ошибка: " + str(err))

        await bot.send_message(message.chat.id,
                               "Что будем делать дальше?", reply_markup=main_menu)

        del photo_buffer[message.chat.id]
        
    # Mone 
    elif photo_buffer[message.chat.id].st_type == 'Mone' and \
            photo_buffer[message.chat.id].need_photos == 1:
        await bot.send_message(message.chat.id, "Происходит обработка изображения")

        try:
            output = monet(photo_buffer[message.chat.id].photos[0])
            await bot.send_document(message.chat.id, deepcopy(output))		
            await bot.send_photo(message.chat.id, output)
			
        except Exception as err:
            await bot.send_message(message.chat.id,
                                   "Произошла ошибка. Сообщение об ошибке отправлено создателю бота.")
            await bot.send_message(ERROR_ID, "Произошла ошибка: " + str(err))

        await bot.send_message(message.chat.id,
                               "Что будем делать дальше?", reply_markup=main_menu)
        del photo_buffer[message.chat.id]

def style_transfer(content,styl):
    output, ratio = style(content,styl)

    return tensor2img(output, ratio)

def monet(image):
    output, ratio= monet_gan(image)
	
    return tensor2img(output, ratio)

def face_man(image):
    output, ratio = face_gan_man(image)
	
    return tensor2img(output, ratio)

def face_woman(image):
    output, ratio = face_gan_woman(image)
	
    return tensor2img(output, ratio)

def tensor2img(tensor, ratio):
    output = tensor.squeeze(0).permute(1, 2, 0) * 255
    output = Image.fromarray(output.type(torch.uint8).numpy())
    output = output.resize((int(512 * ratio),512), Image.ANTIALIAS)


    to_bytes = BytesIO()
    to_bytes.name = 'result.jpeg'
    output.save(to_bytes, 'JPEG')
    to_bytes.seek(0)

    return to_bytes

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
