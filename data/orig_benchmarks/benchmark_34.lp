d1(o4).
d0(o4).
d6(o4).
d6(o5).
d8(o4).
d5(o3).

d4(X) :- d6(X), d7(X), d1(X).
d3(X) :- d6(X), d5(X), d1(X).
d0(X) :- d2(X), d7(X), d8(X).
d2(X) :- d0(X), d5(X), d8(X).
d7(X) :- d3(X), d6(X), d5(X).
d5(X) :- d0(X), d6(X), d8(X).
d9(X) :- d0(X), d7(X), d1(X).
d8(X) :- d2(X), d3(X), d7(X).
d0(X) :- d3(X), d7(X), d1(X).
d0(X) :- d2(X), d3(X), d7(X).
d9(X) :- d2(X), d7(X), d4(X).