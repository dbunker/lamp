d7(o0).
d7(o9).
d9(o9).
d4(o9).
d0(o9).
d8(o5).
d8(o0).
d9(o6).
d0(o2).

d5(X) :- d9(X), d6(X).
d1(X) :- d9(X), d5(X).
d5(X) :- d3(X), d7(X).
d0(X) :- d7(X), d1(X).
d4(X) :- d9(X), d1(X).
d0(X) :- d5(X), d7(X).
d7(X) :- d0(X), d5(X).
d4(X) :- d5(X), d1(X).
d6(X) :- d9(X), d5(X).
d3(X) :- d9(X), d4(X).
d4(X) :- d0(X), d7(X).