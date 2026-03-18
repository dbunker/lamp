d9(o5).
d5(o5).
d4(o0).
d4(o8).
d1(o6).
d0(o6).
d1(o3).
d0(o3).
d7(o5).

d8(X) :- d9(X), d5(X), d4(X).
d9(X) :- d0(X), d8(X), d1(X).
d2(X) :- d7(X), d8(X), d4(X).
d8(X) :- d9(X), d5(X), d7(X).
d6(X) :- d0(X), d2(X), d4(X).
d0(X) :- d3(X), d7(X), d6(X).
d9(X) :- d0(X), d3(X), d1(X).
d3(X) :- d6(X), d7(X), d4(X).
d4(X) :- d9(X), d5(X), d8(X).
d1(X) :- d9(X), d5(X), d8(X).
d9(X) :- d2(X), d3(X), d5(X).