d3(o1).
d1(o5).
d3(o7).
d1(o9).
d5(o4).
d4(o4).
d1(o8).
d0(o9).
d5(o6).

d0(X) :- d5(X), not d4(X).
d0(X) :- d3(X).
d2(X) :- d3(X).
d1(X) :- d4(X).