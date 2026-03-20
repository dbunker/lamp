d8(o2).
d5(o5).
d0(o0).
d4(o0).
d5(o4).
d2(o4).
d9(o2).
d5(o2).
d7(o5).

d2(X) :- d5(X), d7(X).
d0(X) :- d2(X), not d8(X).
d9(X) :- d5(X).
d8(X) :- d7(X).
d3(X) :- d7(X).
d3(X) :- d4(X).
d1(X) :- d8(X).
d1(X) :- d7(X).