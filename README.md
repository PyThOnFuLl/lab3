# lab3 | РАЗРАБОТКА СИСТЕМЫ МАШИННОГО ОБУЧЕНИЯ
Отчет по лабораторной работе #3 выполнил(а):
- Данилов Максим Андреевич
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Разработать и понять как устроена данная система машинного обучения на MLAgent

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity. При выполнении задания можно использовать видео- материалы и исходные данные, предоставленные преподавателями курса.
Ход работы:
- Создайте на сцене плоскость, куб и сферу так, как показано на рисунке ниже. Создайте простой C# скрипт-файл и подключите его к сфере.
![image](https://user-images.githubusercontent.com/100462831/196674708-ce840a24-8b17-4383-96e8-57fb6bc47ef5.png)



- Присвойте сфере компоненты и параметры и привяжите скрипт к сфере:  
![image](https://user-images.githubusercontent.com/100462831/196675247-531c3bc7-fba4-4b58-8487-00f98a52a576.png)
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```


- В корень проекта добавьте файл конфигурации нейронной сети.

```yaml
behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```

- Проверьте работоспособность проекта.

![image](https://user-images.githubusercontent.com/100462831/196676105-28cfed26-7c81-4839-8914-e72e314a4176.png)
![image](https://user-images.githubusercontent.com/100462831/196676326-bab73fbe-92ff-4fb4-b180-8eb7b3ab1307.png)


Вывод: Куб появляется на арене в рандомных точках. Процесс воспроизводится циклично.

## Задание 2
### Детально описать конфигурационный файл для машинного обучения  

```yaml
behaviors: #cоздание списка "Модель поведения" дял разных агентов
  RollerBall: #cоздание списка конкретного объекта
    trainer_type: ppo #ppo - это алгоритм обучения с подкреплением от OpenAi
    hyperparameters:
      batch_size: 10 #количество опыта на каждоый итерации
      buffer_size: 100 #колличество опыта которое необхдимо собрать для перехода к обучению или изучении модели
      learning_rate: 3.0e-4 #скорость обучения
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear #определение изменения скорости обучения с течением времени
    network_settings:
      normalize: false #к входным данным не применяется нормализация
      hidden_units: 128 #количество нейронов в слоях нейронной сети
      num_layers: 2 #количество слоев нейроной сети
    reward_signals: #настройка сигналов вознаграждения
      extrinsic:
        gamma: 0.99
        strength: 1.0 #коэффициент вознаграждения
    max_steps: 500000 #количество шагов для завершения обучения
    time_horizon: 64 #количество шагов для добавления в буфер опыта
    summary_freq: 10000 #количество шагов для создавния и вывода статистикиобучения
```

- Компонент Decision Requester запрашивает процесс принятия решения для агента через разные временные промежутки, вызывая функцию RequestDecision().

Decision Period - параметр, который определяет частоту, с которой агент запрашивает решение. N - период принятия решеняи. Он означает, что Агент будет запрашивать решение каждые n шагов обучения.


- Behavior Parameters - компонент, который определяет, как именно объект принимает решения.

Behavior Name - имя текущего поведения, которое используется в качестве базового имени и указывается в файле конфигурации модели.

Behavior Type - параметр который определяет, какой тип поведения будет использовать Агент. Default - Агент будет использовать удаленный процесс обучения, запущенный через python для принятия решений. InferenceOnly агент всегда будет использовать предоставленную моделью нейронной сети. HeuristicOnly - всегда используется эвристический метод.

Model - это используемая модель нейронной сети.

InferenceDevice - это выбор между CPU и GPU для предоставленной модели нейронной сети.

Vector Observation - это вектор чисел с плавающей запятой, которые содержат информацию для принятия агентом решений. Вектор заполняется в функции CollectObservations.

Actions - это инструкции в форме действий. Действия делятся на два типа: непрерывные и дискретные.


## Задание 3
### Доработать сцену и обучить MLAgent так, чтобы шар проходил между двумя кубами. Кубы случайным образом меняют свои координаты на плоскости.


## Выводы
Баланс в играх - равновесие различных показателей игровых объектов или персонажей при помощи всевозможных методик рассчета. Баланс можно прощупать только при помощи опыта пользователей и постоянных тестов. Система машинного обучения (MLAgent) помогает достичь хорошего игрового баланса. Например, баланс в характеристиках игровых персонажей и/или экономика.


| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
