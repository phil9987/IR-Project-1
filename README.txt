The programm assumes that the codes are stored in src/main/resources

src/main/
├── java
├── resources
│   ├── codes
│   └── data
│       ├── test
│       │   └── test.zip
│       ├── train
│       │   └── train.zip
│       └── validation
│           └── validation.zip
└── scala

Additionally tinyir.jar must be placed in lib/
Once this is done the program can be ran with

sbt run

you can then choose which classifier to run.