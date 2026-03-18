d9(o0).
d0(o5).
d5(o1).
d0(o1).
d3(o4).
d7(o3).

d8(X) :- not d0(X), d7(X).
d7(X) :- not d5(X), d0(X).
d2(X) :- d5(X).
d6(X) :- d5(X).