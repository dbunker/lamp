d1(o4).
d5(o5).
d0(o5).
d2(o1).
d3(o5).
d0(o3).

d1(X) :- d3(X), d8(X).
d7(X) :- d9(X), d4(X).
d6(X) :- d2(X), d8(X).
d7(X) :- d8(X), d1(X).
d4(X) :- d0(X), d3(X).
d0(X) :- d6(X), d4(X).
d8(X) :- d7(X), d1(X).
d4(X) :- d0(X), d5(X).
d9(X) :- d3(X), d4(X).
d6(X) :- d1(X), d8(X).
d7(X) :- d9(X), d1(X).