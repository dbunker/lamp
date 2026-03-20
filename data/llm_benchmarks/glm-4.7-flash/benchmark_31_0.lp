d8(o2).
d7(o0).
d9(o5).
d4(o0).
d5(o4).
d6(o2).

d6(X) :- not d5(X), d7(X).
d1(X) :- not d8(X), d9(X).
d7(X) :- d8(X).
d9(X) :- d8(X).
d8(X) :- d5(X).