d4(o5).
d3(o6).
d4(o4).
d6(o3).
d4(o8).
d0(o2).

d1(X) :- d3(X).
d8(X) :- d6(X).
d7(X) :- d3(X).
d5(X) :- d0(X).
d2(X) :- not d0(X), d4(X).
d4(X) :- d0(X).
d4(X) :- d3(X).