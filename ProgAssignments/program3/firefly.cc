/************************************************************
                        Course          :       CSC506
                        Source          :       msi.cc
                        Instructor      :       Ed Gehringer
                        Email Id        :       efg@ncsu.edu
------------------------------------------------------------
        © Please do not replicate this code without consulting
                the owner & instructor! Thanks!
*************************************************************/
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
using namespace std;
#include "main.h"
#include "firefly.h"


/*
------------------------------------------------------
Don't modify the fucntions included in this section
------------------------------------------------------
*/
cache_line * Firefly::allocate_line_dir(ulong addr) {
	return NULL;
}
void Firefly::signalRd(ulong addr, int processor_number){
}
void Firefly::signalRdX(ulong addr, int processor_number){
}
void Firefly::signalUpgr(ulong addr, int processor_number){
}
void Firefly::Inv(ulong addr) {
}
void Firefly::Int(ulong addr) {
}
void Firefly::PrRdDir(ulong addr, int processor_number) {
}
void Firefly::PrWrDir(ulong addr, int processor_number) {
}
//-------------------------------------------------------------
//Section ends here. edit the functions in the section below 
//-------------------------------------------------------------

void Firefly::PrRd(ulong addr, int processor_number) 
{
}

void Firefly::PrWr(ulong addr, int processor_number) 
{
}


void Firefly::BusRd(ulong addr)
{
}


void Firefly::BusRdX(ulong addr) 
{
}


void Firefly::BusUpgr(ulong addr) 
{
}

void Firefly::BusWr(ulong addr) 
{
}

bool Firefly::writeback_needed(cache_state state) 
{
}

