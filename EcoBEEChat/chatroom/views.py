from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.views.generic.edit import CreateView
from django.views.decorators.csrf import csrf_exempt

from .models import Room
from .models import UserInputDataset

from .forms import DatasetPostForm

import json

import random
import tensorflow as tf
import os
import sys


###### import for transformer model inferencing ######
from .transformer import data as tranformer_data
from .transformer import model as transformer
from .transformer.configs import DEFINES as DEFINES_transformer

###### import for seq2seq model inferencing #######
from .seq2seq import data as seq2seq_data
from .seq2seq import model as seq2seq
from .seq2seq.configs import DEFINES as DEFINES_seq2seq


###### import for attention_seq2seq model inferencing ######
from .attention_seq2seq import data as attention_seq2seq_data
from .attention_seq2seq import model as attention_seq2seq
from .attention_seq2seq.configs import DEFINES as DEFINES_attention_seq2seq



def index(request):
    chatroom_list = Room.objects.order_by('rank')[:5]
    context = {
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/index.html', context)


def room(request, lm_name):
    chatroom_list = Room.objects.order_by('rank')[:5]

    context = {
        'lm_name': mark_safe(json.dumps(lm_name)),
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/room.html', context)


def detail(request, lm_name):

    return HttpResponse("You're looking at chatroom using %s." % lm_name)


@csrf_exempt
def post_dataset(request, lm_name):

    question = request.POST.get('question', None)
    answer = request.POST.get('answer', None)


    new_dataset = UserInputDataset(question=question, answer=answer)       # save to DB
    new_dataset.save()

    print("success to insert new QA dataset from the user")
    print("Q: " + question + " || A: " + answer)

    data = {
        'is_valid': 1
    }

    return JsonResponse(data)



def message(request, message, lm_name):
    # if lm_name == 'tranformer':
    # if lm_name == 'seq2seq':
    # if lm_name == 'bert':
    #return HttpResponse("answer: %s" % (message))

    if "너" in message and "이름" in message:
        return HttpResponse("%s" % ("제 이름은 EcoBEE chat이에요 :)"))


    if lm_name == 'transformer':
        print(lm_name)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        arg_length = len(sys.argv)

        if (arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        # Construct vocab file
        char2idx, idx2char, vocabulary_length = tranformer_data.load_vocabulary()

        ################# Encoder process #################
        input = message

        print(input)
        predic_input_enc, predic_input_enc_length = tranformer_data.enc_processing([input], char2idx)
        # Since it is not train process, there no input to decoder.
        # "" (empty string) is delivered just for fitting in the fixed structure
        predic_output_dec, predic_output_dec_length = tranformer_data.dec_output_processing([""], char2idx)
        # Since it is not train process, there no output from decoder.
        # "" (empty string) is delivered just for fitting in the fixed structure
        predic_target_dec = tranformer_data.dec_target_processing([""], char2idx)


        # Construct tf estimator
        transformer_classifier = tf.estimator.Estimator(
            model_fn=transformer.Model,
            model_dir=DEFINES_transformer.check_point_path_transformer,
            params={  # Parameter passing to model
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                'model_hidden_size': DEFINES_transformer.model_hidden_size_transformer,
                'ffn_hidden_size': DEFINES_transformer.ffn_hidden_size_transformer,
                'attention_head_size': DEFINES_transformer.attention_head_size_transformer,
                'learning_rate': DEFINES_transformer.learning_rate_transformer,
                'vocabulary_length': vocabulary_length,
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                'layer_size': DEFINES_transformer.layer_size_transformer,
                'max_sequence_length': DEFINES_transformer.max_sequence_length_transformer,
                'xavier_initializer': DEFINES_transformer.xavier_initializer_transformer
            })


        answer = ""

        for i in range(25):
            if i > 0:
                predic_output_dec, predic_output_decLength = tranformer_data.dec_output_processing([answer], char2idx)
                predic_target_dec = tranformer_data.dec_target_processing([answer], char2idx)

            ########### Get predicted answer ##########
            predictions = transformer_classifier.predict(
                input_fn=lambda: tranformer_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

            answer, finished = tranformer_data.pred_next_string(predictions, idx2char)

            if finished:
                break

        print("answer: ", answer)



    elif lm_name == 'seq2seq':
        print(lm_name)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        arg_length = len(sys.argv)

        if(arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        char2idx,  idx2char, vocabulary_length = seq2seq_data.load_vocabulary()
        input = message

        print(input)

        predic_input_enc, predic_input_enc_length = seq2seq_data.enc_processing([
                                                                        input], char2idx)
        predic_output_dec, predic_output_dec_length = seq2seq_data.dec_input_processing([
                                                                                ""], char2idx)
        predic_target_dec = seq2seq_data.dec_target_processing([""], char2idx)

        classifier = tf.estimator.Estimator(
            model_fn=seq2seq.model,
            model_dir=DEFINES_seq2seq.check_point_path_seq2seq,
            params={
                'hidden_size': DEFINES_seq2seq.hidden_size_seq2seq,
                'layer_size': DEFINES_seq2seq.layer_size_seq2seq,
                'learning_rate': DEFINES_seq2seq.learning_rate_seq2seq,
                'vocabulary_length': vocabulary_length,
                'embedding_size': DEFINES_seq2seq.embedding_size_seq2seq,
                'embedding': DEFINES_seq2seq.embedding_seq2seq,
                'multilayer': DEFINES_seq2seq.multilayer_seq2seq,
            })

        predictions = classifier.predict(
            input_fn=lambda: seq2seq_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, DEFINES_seq2seq.batch_size_seq2seq))

        answer = seq2seq_data.pred2string(predictions, idx2char)


    elif lm_name == 'attention_seq2seq':
        print(lm_name)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        arg_length = len(sys.argv)

        if(arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        char2idx,  idx2char, vocabulary_length = attention_seq2seq_data.load_vocabulary()

        input = message

        print(input)
        predic_input_enc, predic_input_enc_length = attention_seq2seq_data.enc_processing([
                                                                                          input], char2idx)

        predic_target_dec, _ = attention_seq2seq_data.dec_target_processing([
                                                                            ""], char2idx)

        if DEFINES_attention_seq2seq.serving_attention_seq2seq == True:

            predictor_fn = tf.contrib.predictor.from_saved_model(
                export_dir="/home/evo_mind/DeepLearning/NLP/Work/ChatBot2_Final/data_out/model/1541575161"
            )
        else:
            # Construct estimator
            classifier = tf.estimator.Estimator(
                model_fn=attention_seq2seq.Model,
                model_dir=DEFINES_attention_seq2seq.check_point_path_attention_seq2seq,
                params={
                    'hidden_size': DEFINES_attention_seq2seq.hidden_size_attention_seq2seq,
                    'layer_size': DEFINES_attention_seq2seq.layer_size_attention_seq2seq,
                    'learning_rate': DEFINES_attention_seq2seq.learning_rate_attention_seq2seq,
                    'teacher_forcing_rate': DEFINES_attention_seq2seq.teacher_forcing_rate_attention_seq2seq, # 학습시 디코더 인풋 정답 지원율 설정
                    'vocabulary_length': vocabulary_length,
                    'embedding_size': DEFINES_attention_seq2seq.embedding_size_attention_seq2seq,
                    'embedding': DEFINES_attention_seq2seq.embedding_attention_seq2seq,
                    'multilayer': DEFINES_attention_seq2seq.multilayer_attention_seq2seq,
                    'attention': DEFINES_attention_seq2seq.attention_attention_seq2seq,
                    'teacher_forcing': DEFINES_attention_seq2seq.teacher_forcing_attention_seq2seq,
                    'loss_mask': DEFINES_attention_seq2seq.loss_mask_attention_seq2seq, # PAD에 대한 마스크를 통해 loss를 제한
                    'serving': DEFINES_attention_seq2seq.serving_attention_seq2seq
                })

        if DEFINES_attention_seq2seq.serving_attention_seq2seq == True:
            predictions = predictor_fn({'input':predic_input_enc, 'output':predic_target_dec})

        else:
            predictions = classifier.predict(
                input_fn=lambda: attention_seq2seq_data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES_attention_seq2seq.batch_size_attention_seq2seq))

        # convert indexed sentence into string sentence
        answer = attention_seq2seq_data.pred2string(predictions, idx2char)

    else:
        answer = "죄송합니다. 아직 서비스 준비중 입니다. 빠른 시일 내에 완성하겠습니다."


    return HttpResponse("%s" % answer)


# this is only for test
'''
def helloworld(request, num, lm_name):
    n1 = int(num)
    n2 = 1
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)

    add = tf.add(x, y)
    with tf.Session() as sess:
        z = sess.run(add, feed_dict={x:n1, y:n2})

    return HttpResponse("sum of %s and 1 is %s" %(num, z))
'''
