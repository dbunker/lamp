d1(o4).
d3(o6).
d8(o3).
d4(o0).
d4(o1).
d8(o0).
d8(o4).
d6(o9).
d7(o5).

d6(X) :- d0(X), d5(X).
d2(X) :- d3(X), d7(X).
d3(X) :- d1(X), d4(X).
d5(X) :- d6(X), d8(X).
d2(X) :- d6(X), d5(X).
d9(X) :- d7(X), d8(X).
d7(X) :- d0(X), d2(X).
d4(X) :- d1(X), d8(X).
d4(X) :- d2(X), d5(X).
d8(X) :- d7(X), d1(X).
d0(X) :- d8(X), d4(X).