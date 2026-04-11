d4(o0).
d4(o1).
d1(o5).
d0(o5).
d2(o1).

d5(X) :- d2(X).
d5(X) :- d1(X).
d3(X) :- not d1(X), d4(X).
d4(X) :- d1(X).